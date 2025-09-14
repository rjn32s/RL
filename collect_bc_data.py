import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pygame
import pickle
from datetime import datetime
import os
import random

# --- Key to Action mapping ---
KEY_ACTIONS = {
    pygame.K_w: np.array([-0.1, 0.0, 0.0]),     # move +x (forward)
    pygame.K_s: np.array([0.1, 0.0, 0.0]),  # move -x (backward)
    pygame.K_a: np.array([0.0, -0.1, 0.0]),   # move +y (left)
    pygame.K_d: np.array([0.0, 0.1, 0.0]), # move -y (right)
    pygame.K_z: np.array([0.0, 0.0, 0.1]),      # move +z (up)
    pygame.K_x: np.array([0.0, 0.0, -0.1]),     # move -z (down)
}

def init_env():
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlaceDense-v4", render_mode="human", max_episode_steps=5000)
    return env

def collect_bc_dataset(num_episodes=15, sample_freq=5):
    """
    Collect multiple successful episodes for Behavioral Cloning training
    
    Args:
        num_episodes: Number of successful episodes to collect
        sample_freq: Sampling frequency for data collection
    """
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("BC Data Collection - Pick and Place Task")
    
    print(f"\n🎯 Behavioral Cloning Data Collection")
    print(f"📊 Target: {num_episodes} successful episodes")
    print(f"⚡ Sampling: Every {sample_freq} steps")
    
    # Print controls
    print("\n=== CONTROLS ===")
    print("WASD: Move forward/backward/left/right")
    print("Z/X: Move up/down (z-axis)")
    print("O/C: Open/Close gripper")
    print("ESC: Quit current episode")
    print("================\n")

    clock = pygame.time.Clock()
    successful_episodes = 0
    total_attempts = 0
    all_episodes = []

    # Create output folder for BC data
    output_folder = "bc_dataset"
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Saving episodes to: {output_folder}/")

    # Get seed range from user
    print(f"\n🌱 Seed Configuration:")
    print("Enter seed range for diverse episodes (e.g., 100-200 for seeds 100, 101, 102...)")
    start_seed = int(input("Start seed (default=100): ") or "100")
    end_seed = int(input("End seed (default=300): ") or "300")
    
    if end_seed - start_seed < num_episodes:
        end_seed = start_seed + num_episodes
        print(f"⚠️  Adjusted end seed to {end_seed} to ensure enough seeds")

    print(f"✅ Using seeds {start_seed} to {end_seed}")

    while successful_episodes < num_episodes:
        total_attempts += 1
        current_seed = start_seed + (successful_episodes % (end_seed - start_seed))
        
        print(f"\n🎯 Episode {successful_episodes + 1}/{num_episodes} (Attempt {total_attempts})")
        print(f"🌱 Using seed: {current_seed}")
        
        # Create fresh environment for each episode to avoid hanging
        print("🔄 Creating fresh environment...")
        env = init_env()
        obs, info = env.reset(seed=current_seed)
        done = False
        step_count = 0
        episode_data = []
        
        # Log initial positions
        initial_object_pos = obs["achieved_goal"]
        initial_target_pos = obs["desired_goal"]
        print(f"Object: [{initial_object_pos[0]:.3f}, {initial_object_pos[1]:.3f}, {initial_object_pos[2]:.3f}]")
        print(f"Target: [{initial_target_pos[0]:.3f}, {initial_target_pos[1]:.3f}, {initial_target_pos[2]:.3f}]")
        print(f"Distance: {np.linalg.norm(initial_object_pos - initial_target_pos):.3f}")
        
        while not done:
            action = np.zeros(4, dtype=np.float32)
            action_taken = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("Episode terminated by user")
                    done = True
                    break

            keys = pygame.key.get_pressed()
            
            # Handle movement controls
            for key, delta in KEY_ACTIONS.items():
                if keys[key]:
                    if key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_z, pygame.K_x]:
                        action[:3] = delta[:3]
                        action_taken = True
                        break
            
            # Handle gripper controls
            if keys[pygame.K_c]:  # Close gripper
                action[3] = -1.0
                action_taken = True
            elif keys[pygame.K_o]:  # Open gripper
                action[3] = 1.0
                action_taken = True

            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Progress tracking
            if step_count % 50 == 0:
                current_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
                print(f"Step {step_count}, Distance: {current_distance:.3f}, Reward: {reward:.3f}")
            
            # Save transition data
            if step_count % sample_freq == 0 or terminated or truncated:
                transition = {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": terminated or truncated,
                    "info": info,
                    # BC-specific data
                    "object_pos": obs["achieved_goal"].copy(),
                    "target_pos": obs["desired_goal"].copy(),
                    "robot_state": obs["observation"].copy(),
                    "next_object_pos": next_obs["achieved_goal"].copy(),
                    "step_count": step_count,
                    "sampling_freq": sample_freq
                }
                episode_data.append(transition)

            obs = next_obs
            env.render()
            clock.tick(20)

            # Check for success
            episode_successful = False
            current_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            
            if current_distance < 0.05:  # Success threshold
                print(f"🎉 SUCCESS! Distance: {current_distance:.4f}")
                episode_successful = True
                terminated = True
                done = True
            elif reward > -0.05:  # High reward
                print(f"🎉 SUCCESS! Reward: {reward:.3f}")
                episode_successful = True
                terminated = True
                done = True
            elif terminated or truncated:
                done = True
                if terminated:
                    episode_successful = True
                    print(f"🎉 Episode completed! Steps: {step_count}")
                else:
                    print(f"⏰ Time limit reached. Steps: {step_count}")
                print(f"Final distance: {current_distance:.4f}")
            
            # Handle episode end
            if done:
                if episode_successful:
                    successful_episodes += 1
                    print(f"✅ Episode {successful_episodes}/{num_episodes} completed!")
                    print(f"📊 Collected {len(episode_data)} transitions")
                    
                    # Add episode metadata
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    episode_data.append({
                        "episode_metadata": {
                            "episode_id": f"bc_ep_{successful_episodes:02d}_{ts}",
                            "episode_number": successful_episodes,
                            "total_steps": step_count,
                            "total_transitions": len(episode_data),
                            "sampling_freq": sample_freq,
                            "success": True,
                            "final_reward": reward,
                            "final_distance": current_distance,
                            "seed": current_seed
                        }
                    })
                    
                    # Save individual episode
                    episode_file = f"{output_folder}/bc_episode_{successful_episodes:02d}_{ts}.pkl"
                    with open(episode_file, "wb") as f:
                        pickle.dump(episode_data, f)
                    print(f"💾 Saved: {episode_file}")
                    
                    # Add to all episodes
                    all_episodes.extend(episode_data)
                    
                    if successful_episodes < num_episodes:
                        print(f"🎯 Ready for next episode! ({num_episodes - successful_episodes} remaining)")
                        # Close environment before next episode
                        env.close()
                        input("Press Enter to continue...")
                else:
                    print(f"❌ Episode failed. Trying again...")
                    # Close environment before retry
                    env.close()
                    print("Press Enter to retry...")
                    input()
                
                pygame.time.wait(1000)

    # Final cleanup
    try:
        env.close()
    except:
        pass
    pygame.quit()

    # Save combined dataset
    combined_file = f"{output_folder}/bc_combined_dataset.pkl"
    with open(combined_file, "wb") as f:
        pickle.dump(all_episodes, f)
    
    print(f"\n🎉 BC Data Collection Complete!")
    print(f"📁 Individual episodes: {output_folder}/")
    print(f"📁 Combined dataset: {combined_file}")
    print(f"✅ Successful episodes: {successful_episodes}")
    print(f"🔄 Total attempts: {total_attempts}")
    print(f"📊 Total transitions: {len(all_episodes)}")
    print(f"💡 Ready for BC training!")

    return all_episodes

if __name__ == "__main__":
    print("=== Behavioral Cloning Data Collection ===")
    print("This will collect multiple successful episodes for BC training")
    
    # Get user preferences
    num_episodes = int(input("Number of episodes to collect (default=15): ") or "15")
    sample_freq = int(input("Sampling frequency (default=5): ") or "5")
    
    print(f"\nThis will collect {num_episodes} episodes with sampling every {sample_freq} steps")
    print("Each episode must be successful to be saved!")
    print("If you fail, just try again - only success counts! 🎯")
    print("\n💡 Note: Environment will be recreated for each episode to prevent hanging")
    
    input("\nPress Enter to start data collection...")
    
    collect_bc_dataset(num_episodes, sample_freq)
