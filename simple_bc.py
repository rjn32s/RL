import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime

class SimpleBCDataset(Dataset):
    """Simple dataset for BC training"""
    
    def __init__(self, transitions):
        self.transitions = transitions
        
        # Extract states and actions
        self.states = []
        self.actions = []
        
        for transition in transitions:
            if 'episode_metadata' not in transition:
                state = transition['robot_state'].astype(np.float32)
                action = transition['action'].astype(np.float32)
                
                self.states.append(state)
                self.actions.append(action)
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        
        print(f"📊 Simple BC Dataset:")
        print(f"  States: {self.states.shape}")
        print(f"  Actions: {self.actions.shape}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx]), torch.FloatTensor(self.actions[idx])

class SimpleBCPolicy(nn.Module):
    """Simple but effective BC policy"""
    
    def __init__(self, state_dim=25, action_dim=4):
        super(SimpleBCPolicy, self).__init__()
        
        # Simple 3-layer network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        return self.network(state)
    
    def predict(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            return self.forward(state).squeeze(0).numpy()

def load_bc_data(data_path="bc_dataset/improved_bc_combined_dataset.pkl"):
    """Load BC training data"""
    print(f"📂 Loading BC data from: {data_path}")
    
    with open(data_path, "rb") as f:
        transitions = pickle.load(f)
    
    # Filter out metadata entries
    data_transitions = [t for t in transitions if 'episode_metadata' not in t]
    
    print(f"📊 Loaded {len(data_transitions)} transitions")
    return data_transitions

def train_simple_bc(transitions, epochs=1000, batch_size=32, learning_rate=0.001):
    """Train simple BC policy"""
    
    print(f"\n🤖 Training Simple BC Policy")
    print(f"📊 Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Create dataset
    dataset = SimpleBCDataset(transitions)
    
    # Split data
    train_data, test_data = train_test_split(
        list(range(len(dataset))), 
        test_size=0.2, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_data)
    test_dataset = torch.utils.data.Subset(dataset, test_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📈 Train samples: {len(train_dataset)}")
    print(f"📈 Test samples: {len(test_dataset)}")
    
    # Create model
    model = SimpleBCPolicy(state_dim=25, action_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            
            predicted_actions = model(batch_states)
            loss = criterion(predicted_actions, batch_actions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions in test_loader:
                predicted_actions = model(batch_states)
                loss = criterion(predicted_actions, batch_actions)
                test_loss += loss.item()
        
        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs("simple_bc_models", exist_ok=True)
    model_path = f"simple_bc_models/simple_bc_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': 25,
        'action_dim': 4,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss
    }, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"🏆 Best test loss: {best_test_loss:.6f}")
    
    # Plot training curves
    plot_simple_curves(train_losses, test_losses)
    
    return model, model_path

def plot_simple_curves(train_losses, test_losses):
    """Plot simple training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Simple BC Training Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

def evaluate_simple_bc(model, num_episodes=10):
    """Evaluate simple BC policy"""
    
    print(f"\n🎯 Evaluating Simple BC Policy")
    print(f"📊 Episodes: {num_episodes}")
    
    # Initialize environment
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlaceDense-v4", render_mode="human", max_episode_steps=1000)
    
    results = []
    
    for episode in range(num_episodes):
        seed = 100 + episode
        print(f"\n🎯 Episode {episode + 1}/{num_episodes} (Seed: {seed})")
        
        obs, info = env.reset(seed=seed)
        done = False
        step_count = 0
        total_reward = 0.0
        
        # Log initial positions
        initial_object_pos = obs["achieved_goal"]
        initial_target_pos = obs["desired_goal"]
        initial_distance = np.linalg.norm(initial_object_pos - initial_target_pos)
        
        print(f"Object: [{initial_object_pos[0]:.3f}, {initial_object_pos[1]:.3f}, {initial_object_pos[2]:.3f}]")
        print(f"Target: [{initial_target_pos[0]:.3f}, {initial_target_pos[1]:.3f}, {initial_target_pos[2]:.3f}]")
        print(f"Initial distance: {initial_distance:.3f}")
        
        distances = [initial_distance]
        
        while not done and step_count < 500:  # Shorter episodes for testing
            # Get current state
            current_state = obs["observation"].astype(np.float32)
            
            # Predict action using BC policy
            predicted_action = model.predict(current_state)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(predicted_action)
            
            step_count += 1
            total_reward += reward
            
            # Calculate distance
            current_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            distances.append(current_distance)
            
            # Progress tracking
            if step_count % 50 == 0:
                print(f"  Step {step_count:3d} | Distance: {current_distance:.3f} | Reward: {reward:.3f}")
            
            # Check for success
            if current_distance < 0.05:
                print(f"🎉 SUCCESS! Distance: {current_distance:.4f}")
                success = True
                break
            elif terminated or truncated:
                success = False
                print(f"⏰ Episode ended. Distance: {current_distance:.3f}")
                break
            
            obs = next_obs
            env.render()
        
        # Final evaluation
        final_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        success = final_distance < 0.05
        
        episode_result = {
            'episode': episode + 1,
            'success': success,
            'steps': step_count,
            'total_reward': total_reward,
            'final_distance': final_distance,
            'min_distance': min(distances)
        }
        
        results.append(episode_result)
        
        print(f"📊 Episode {episode + 1} Results:")
        print(f"  Success: {'✅' if success else '❌'}")
        print(f"  Steps: {step_count}")
        print(f"  Final Distance: {final_distance:.3f}")
        print(f"  Min Distance: {min(distances):.3f}")
    
    env.close()
    
    # Calculate overall statistics
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_final_distance = np.mean([r['final_distance'] for r in results])
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"📊 Success Rate: {success_rate:.1%}")
    print(f"📊 Average Steps: {avg_steps:.1f}")
    print(f"📊 Average Reward: {avg_reward:.3f}")
    print(f"📊 Average Final Distance: {avg_final_distance:.3f}")
    
    return results

def main():
    """Main function"""
    print("🤖 Simple BC Training & Evaluation")
    print("=" * 40)
    
    # Load data
    transitions = load_bc_data()
    
    if len(transitions) < 50:
        print(f"⚠️  Warning: Only {len(transitions)} transitions available.")
    
    # Train model
    model, model_path = train_simple_bc(transitions)
    
    print(f"\n✅ Training completed!")
    print(f"💾 Model saved to: {model_path}")
    
    # Evaluate model
    input("\nPress Enter to start evaluation...")
    results = evaluate_simple_bc(model, num_episodes=5)
    
    return model, results

if __name__ == "__main__":
    model, results = main()
