import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pygame
import pickle
from datetime import datetime
import random

# --- Key to Action mapping ---
# action = [dx, dy, dz, gripper]
# Using arrow keys and other non-conflicting keys
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
    # Create environment with very high step limit to allow task completion
    env = gym.make("FetchPickAndPlaceDense-v4", render_mode="human", max_episode_steps=5000)
    return env

def teleop_collect(env, save_file="expert_data.pkl", sample_freq=5, success_only=True):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Keyboard Teleop for Fetch - Pick and Place Task")
    
    # Print controls
    print("\n=== CONTROLS ===")
    print("Arrow Keys: Move forward/backward/left/right")
    print("U/J: Move up/down (z-axis)")
    print("O/C: Open/Close gripper")
    print("ESC: Quit current episode")
    print("================\n")

    dataset = []
    clock = pygame.time.Clock()
    successful_episodes = 0
    total_attempts = 0

    # Create output folder for organized data
    import os
    output_folder = "expert_demos"
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Saving successful episodes to: {output_folder}/")

    # Get seed from user for reproducible episodes
    print(f"\n🌱 Seed Configuration:")
    print("Enter a seed value for reproducible episodes (same seed = same initial positions)")
    print("Examples: 100, 42, 12345, etc.")
    
    while True:
        try:
            seed_input = input("Enter seed value (default=100): ").strip()
            if seed_input == "":
                seed = 100
                break
            else:
                seed = int(seed_input)
                break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    print(f"✅ Using seed: {seed} for all episodes")

    while successful_episodes < 1:  # Only collect 1 successful episode
        total_attempts += 1
        print(f"\n🎯 Attempt {total_attempts} - Goal: Pick up the red object and place it on the green target!")
        print(f"🌱 Using seed: {seed} for reproducible episode")
        obs, info = env.reset(seed=seed)
        done = False
        step_count = 0
        episode_data = []  # Store data for this episode
        
        # Log initial positions for training
        initial_object_pos = obs["achieved_goal"]
        initial_target_pos = obs["desired_goal"]
        print(f"Initial object position: {initial_object_pos}")
        print(f"Target position: {initial_target_pos}")
        print(f"Distance to target: {np.linalg.norm(initial_object_pos - initial_target_pos):.3f}")
        
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
            
            # Handle movement controls - FIXED: Don't accumulate, set directly
            for key, delta in KEY_ACTIONS.items():
                if keys[key]:
                    if key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_z, pygame.K_x]:
                        action[:3] = delta[:3]  # Set movement components directly
                        action_taken = True
                        break  # Only one movement at a time
            
            # Handle gripper controls separately to avoid conflicts
            if keys[pygame.K_c]:  # Close gripper
                action[3] = -1.0  # Close gripper
                action_taken = True
            elif keys[pygame.K_o]:  # Open gripper
                action[3] = 1.0   # Open gripper
                action_taken = True

            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print progress every 10 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}, Reward: {reward:.3f}")
            
            # Debug: Print detailed info every 50 steps and when reward changes significantly
            if step_count % 50 == 0 or abs(reward - (-0.1)) < 0.05:  # When getting close to success
                print(f"DEBUG Step {step_count}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
                if 'is_success' in info:
                    print(f"  is_success={info['is_success']}")
                if 'success' in info:
                    print(f"  success={info['success']}")
                if 'achieved_goal' in info:
                    print(f"  achieved_goal={info['achieved_goal']}")
                if 'desired_goal' in info:
                    print(f"  desired_goal={info['desired_goal']}")
            
            # Debug: Print termination info when it happens
            if terminated or truncated:
                print(f"DEBUG: terminated={terminated}, truncated={truncated}, reward={reward:.3f}")
                if 'is_success' in info:
                    print(f"DEBUG: is_success={info['is_success']}")
                if 'success' in info:
                    print(f"DEBUG: success={info['success']}")
            
            # Save transition with detailed position information (optimized sampling)
            if step_count % sample_freq == 0 or terminated or truncated:
                transition = {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": terminated or truncated,
                    "info": info,
                    # Extract specific position data for training
                    "object_pos": obs["achieved_goal"].copy(),  # Current object position
                    "target_pos": obs["desired_goal"].copy(),  # Target position (constant)
                    "robot_state": obs["observation"].copy(),  # Full robot state
                    "next_object_pos": next_obs["achieved_goal"].copy(),  # Next object position
                    "step_count": step_count,
                    "sampling_freq": sample_freq
                }
                episode_data.append(transition)

            obs = next_obs
            env.render()
            clock.tick(20)  # Increased frame rate for smoother control

            # Check for success detection - improved success criteria
            episode_successful = False
            current_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            
            # Multiple success criteria
            if current_distance < 0.05:  # Object is very close to target
                print(f"🎉 SUCCESS! Object reached target! Distance: {current_distance:.4f}")
                episode_successful = True
                terminated = True
                done = True
            elif reward > -0.05:  # High reward indicates success
                print(f"🎉 Manual success detected! Reward: {reward:.3f}")
                episode_successful = True
                terminated = True
                done = True
            elif terminated or truncated:
                done = True
                if terminated:
                    episode_successful = True
                    print(f"🎉 Episode completed successfully! Task finished in {step_count} steps!")
                else:
                    print(f"⏰ Episode ended due to time limit. Steps: {step_count}")
                print(f"Final reward: {reward:.3f}")
                print(f"Final distance: {current_distance:.4f}")
            
            # If episode ended, check if it was successful
            if done:
                if episode_successful:
                    successful_episodes += 1
                    print(f"✅ SUCCESS! Episode {successful_episodes} completed in {step_count} steps")
                    print(f"📊 Collected {len(episode_data)} transitions")
                    
                    # Add episode metadata BEFORE saving
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    episode_data.append({
                        "episode_metadata": {
                            "episode_id": ts,
                            "total_steps": step_count,
                            "total_transitions": len(episode_data),
                            "sampling_freq": sample_freq,
                            "success": True,
                            "final_reward": reward,
                            "seed": seed  # Save seed for reproducible replay
                        }
                    })
                    
                    # Save this successful episode (individual file only)
                    episode_file = f"{output_folder}/episode_{ts}.pkl"
                    with open(episode_file, "wb") as f:
                        pickle.dump(episode_data, f)
                    print(f"💾 Saved successful episode to: {episode_file}")
                    print(f"🌱 Saved with seed: {seed}")
                    break  # Exit the while loop since we got our successful episode
                else:
                    print(f"❌ Episode failed. Trying again... (Attempt {total_attempts})")
                    print("Press any key to continue to next attempt...")
                    input()
                
                pygame.time.wait(1000)

    env.close()
    pygame.quit()

    print(f"\n🎉 Data collection complete!")
    print(f"📁 Episode saved in: {output_folder}/")
    print(f"✅ Successful episodes: {successful_episodes}")
    print(f"🔄 Total attempts: {total_attempts}")
    print(f"💡 Each episode is saved as a separate .pkl file for easy training!")

if __name__ == "__main__":
    env = init_env()
    
    print("=== Single Success Episode Collection ===")
    print("This will collect ONE successful episode and save it to expert_demos/ folder")
    print("\nSampling frequency options:")
    print("1. Every step (1x) - Most data, slowest")
    print("2. Every 3 steps (3x) - Balanced")
    print("3. Every 5 steps (5x) - Recommended for training")
    print("4. Every 10 steps (10x) - Fastest, less data")
    
    choice = input("Choose sampling frequency (1-4, default=3): ").strip()
    
    if choice == "1":
        sample_freq = 1
        print("Using every step sampling (maximum data)")
    elif choice == "2":
        sample_freq = 3
        print("Using every 3 steps sampling (balanced)")
    elif choice == "4":
        sample_freq = 10
        print("Using every 10 steps sampling (fastest)")
    else:  # default option 3
        sample_freq = 5
        print("Using every 5 steps sampling (recommended for training)")
    
    print(f"\nThis will collect data every {sample_freq} steps + at episode end")
    print("Only SUCCESSFUL episodes will be saved!")
    print("If you fail, just try again - only success counts! 🎯")
    
    teleop_collect(env, sample_freq=sample_freq)
