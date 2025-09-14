import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pickle
import os
import time
from datetime import datetime

def init_env():
    """Initialize the Fetch environment for cloning"""
    gym.register_envs(gymnasium_robotics)
    # Create environment with same settings as data collection
    env = gym.make("FetchPickAndPlaceDense-v4", render_mode="human", max_episode_steps=5000)
    return env

def load_expert_data(episode_file):
    """Load expert demonstration data from pickle file"""
    print(f"📂 Loading expert data from: {episode_file}")
    
    with open(episode_file, "rb") as f:
        episode_data = pickle.load(f)
    
    # Check if there's episode metadata at the end
    if episode_data and isinstance(episode_data[-1], dict) and "episode_metadata" in episode_data[-1]:
        metadata = episode_data[-1]["episode_metadata"]
        print(f"📋 Episode Info:")
        print(f"  Episode ID: {metadata.get('episode_id', 'Unknown')}")
        print(f"  Total Steps: {metadata.get('total_steps', 'Unknown')}")
        print(f"  Success: {metadata.get('success', 'Unknown')}")
        print(f"  Seed: {metadata.get('seed', 'Unknown')}")
        # Remove metadata from data
        transitions = episode_data[:-1]
    else:
        transitions = episode_data
        metadata = {}
    
    print(f"📊 Loaded {len(transitions)} transitions")
    return transitions, metadata

def clone_robot_behavior(env, transitions, target_seed=None, playback_speed=1.0):
    """
    Clone the robot behavior by replaying exact actions from expert data
    
    Args:
        env: Gymnasium environment
        transitions: List of expert transitions
        target_seed: Seed to use for environment reset (if None, uses original seed)
        playback_speed: Speed multiplier for playback (1.0 = real-time, 2.0 = 2x speed)
    """
    print(f"\n🤖 Starting Robot Cloning...")
    print(f"🎯 Target: Replay {len(transitions)} expert actions")
    print(f"⚡ Playback speed: {playback_speed}x")
    
    # Use original seed if available, otherwise use a default
    if target_seed is None:
        target_seed = 42  # Default seed
    
    # Reset environment with the same seed for reproducible initial conditions
    obs, info = env.reset(seed=target_seed)
    
    # Verify positions match (should be identical with same seed)
    expert_initial_pos = transitions[0]['object_pos']
    expert_target_pos = transitions[0]['target_pos']
    current_object_pos = obs['achieved_goal']
    current_target_pos = obs['desired_goal']
    
    print(f"\n🔍 Position Verification (using same seed):")
    print(f"Expert object:  {expert_initial_pos}")
    print(f"Current object: {current_object_pos}")
    print(f"Object match: {'✅' if np.allclose(expert_initial_pos, current_object_pos, atol=0.01) else '❌'}")
    
    print(f"Expert target:  {expert_target_pos}")
    print(f"Current target: {current_target_pos}")
    print(f"Target match: {'✅' if np.allclose(expert_target_pos, current_target_pos, atol=0.01) else '❌'}")
    
    if not (np.allclose(expert_initial_pos, current_object_pos, atol=0.01) and 
            np.allclose(expert_target_pos, current_target_pos, atol=0.01)):
        print(f"\n⚠️  WARNING: Positions don't match despite same seed!")
        print(f"This might indicate a version mismatch or environment issue.")
        print(f"Proceeding anyway - cloning may not work perfectly.")
    
    print(f"🌱 Environment reset with seed: {target_seed}")
    print(f"Initial object position: {obs['achieved_goal']}")
    print(f"Target position: {obs['desired_goal']}")
    print(f"Distance to target: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")
    
    # Calculate delay between actions for smooth playback
    base_delay = 0.05  # 50ms base delay (20 FPS)
    action_delay = base_delay / playback_speed
    
    print(f"\n🎬 Starting action replay...")
    print("=" * 50)
    
    successful_steps = 0
    total_reward = 0.0
    
    for i, transition in enumerate(transitions):
        # Extract the exact action that was performed
        expert_action = transition['action'].copy()
        
        # Execute the action in the environment
        next_obs, reward, terminated, truncated, info = env.step(expert_action)
        
        successful_steps += 1
        total_reward += reward
        
        # Print progress every 25 steps
        if (i + 1) % 25 == 0 or i < 5 or i >= len(transitions) - 5:
            print(f"Step {i+1:3d}/{len(transitions)} | Action: {expert_action} | Reward: {reward:.3f} | Total: {total_reward:.3f}")
            
            # Show position tracking
            current_pos = next_obs['achieved_goal']
            target_pos = next_obs['desired_goal']
            distance = np.linalg.norm(current_pos - target_pos)
            print(f"         Object: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] | Distance: {distance:.3f}")
        
        # Check for early termination
        if terminated or truncated:
            print(f"\n🏁 Episode ended at step {i+1}")
            print(f"   Terminated: {terminated}")
            print(f"   Truncated: {truncated}")
            print(f"   Final reward: {reward:.3f}")
            break
        
        # Render the environment
        env.render()
        
        # Control playback speed
        time.sleep(action_delay)
        
        # Update observation for next iteration
        obs = next_obs
    
    print("=" * 50)
    print(f"🎉 Cloning Complete!")
    print(f"✅ Successfully executed {successful_steps} steps")
    print(f"💰 Total reward: {total_reward:.3f}")
    print(f"📊 Average reward per step: {total_reward/successful_steps:.3f}")
    
    # Final success check
    final_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    print(f"🎯 Final distance to target: {final_distance:.3f}")
    
    if final_distance < 0.05:  # Success threshold
        print("🏆 TASK COMPLETED SUCCESSFULLY! 🏆")
    else:
        print("⚠️  Task not completed - object not close enough to target")
    
    return {
        'successful_steps': successful_steps,
        'total_reward': total_reward,
        'final_distance': final_distance,
        'success': final_distance < 0.05
    }

def main():
    """Main function to run robot cloning"""
    print("🤖 Robot Arm Cloning System")
    print("=" * 40)
    
    # Find the most recent expert episode
    expert_demos_dir = "expert_demos"
    if not os.path.exists(expert_demos_dir):
        print(f"❌ Expert demos directory not found: {expert_demos_dir}")
        return
    
    episode_files = [f for f in os.listdir(expert_demos_dir) if f.endswith('.pkl')]
    if not episode_files:
        print(f"❌ No episode files found in {expert_demos_dir}/")
        return
    
    # Use the most recent episode
    latest_episode = sorted(episode_files)[-1]
    episode_path = os.path.join(expert_demos_dir, latest_episode)
    
    # Load expert data
    transitions, metadata = load_expert_data(episode_path)
    
    if not transitions:
        print("❌ No transitions found in expert data")
        return
    
    # Initialize environment
    env = init_env()
    
    try:
        # Get user preferences
        print(f"\n⚙️  Cloning Configuration:")
        
        # Playback speed selection
        print("Playback speed options:")
        print("1. Real-time (1.0x) - Watch every action")
        print("2. Fast (2.0x) - Double speed")
        print("3. Very fast (5.0x) - Quick replay")
        print("4. Instant (10.0x) - Skip to end")
        
        speed_choice = input("Choose speed (1-4, default=1): ").strip()
        speed_map = {"1": 1.0, "2": 2.0, "3": 5.0, "4": 10.0}
        playback_speed = speed_map.get(speed_choice, 1.0)
        
        # Use the exact seed from the expert data
        original_seed = metadata.get('seed', 42)
        print(f"\n📋 Expert episode used seed: {original_seed}")
        print(f"✅ Using exact same seed for perfect reproducibility")
        target_seed = original_seed
        
        print(f"\n🚀 Starting cloning with:")
        print(f"   Episode: {latest_episode}")
        print(f"   Speed: {playback_speed}x")
        print(f"   Seed: {target_seed}")
        print(f"   Actions: {len(transitions)}")
        
        input("\nPress Enter to start cloning...")
        
        # Run the cloning
        results = clone_robot_behavior(env, transitions, target_seed, playback_speed)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"cloning_results_{timestamp}.txt"
        
        with open(results_file, "w") as f:
            f.write(f"Robot Cloning Results - {timestamp}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Source episode: {latest_episode}\n")
            f.write(f"Playback speed: {playback_speed}x\n")
            f.write(f"Environment seed: {target_seed}\n")
            f.write(f"Successful steps: {results['successful_steps']}\n")
            f.write(f"Total reward: {results['total_reward']:.3f}\n")
            f.write(f"Final distance: {results['final_distance']:.3f}\n")
            f.write(f"Success: {results['success']}\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
    finally:
        env.close()
        print("\n👋 Environment closed. Cloning session complete!")

if __name__ == "__main__":
    main()
