import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.tactile_env import TactileEnv
from train import make_env

def visualize_model(model_path, config_path, num_episodes=5, max_steps=300, epsilon_greedy=0.0, use_heuristic=False, corrupt_pc=False, object_model=None):
    """
    Visualize a trained model in MuJoCo GUI and generate plots with diagnostics.
    Args:
        model_path: Path to the saved model
        config_path: Path to the config file
        num_episodes: Number of episodes to visualize
        max_steps: Max steps per episode
        epsilon_greedy: Probability of random action
        use_heuristic: Use heuristic baseline instead of RL
        corrupt_pc: Corrupt point cloud for reward sanity check
        object_model: Optionally override object model in config
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if object_model is not None:
        config['environment']['model_path'] = object_model

    # Create vectorized environment with rendering
    env = DummyVecEnv([make_env(config, 0, render_mode='human')])

    # Load model with environment
    model = PPO.load(model_path, env=env)

    # Lists to store episode data
    episode_rewards = []
    episode_lengths = []
    tactile_readings = []
    point_clouds = []
    reward_breakdown = []
    chamfer_deltas = []
    occupancy_gains = []
    action_stats = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        prev_pc = None
        prev_voxel_count = 0
        actions_this_ep = []
        rewards_this_ep = []
        chamfers_this_ep = []
        occ_this_ep = []
        tactile_this_ep = []
        pc_this_ep = []
        for step in range(max_steps):
            # Unvectorize obs for logging
            obs_dict = {k: np.array(v[0], dtype=np.float32) for k, v in obs.items()}
            print(f"[Step {step}] Tactile: {obs_dict['tactile']}")
            print(f"[Step {step}] Robot pose: {obs_dict['robot_pose']}")
            print(f"[Step {step}] Prev action: {obs_dict['prev_action']}")
            # Epsilon-greedy or heuristic
            if use_heuristic:
                action = heuristic_policy(obs_dict)
            elif np.random.rand() < epsilon_greedy:
                action = env.action_space.sample()[0]
            else:
                action, _ = model.predict(obs_dict, deterministic=True)
            actions_this_ep.append(action)
            # Step
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            episode_length += 1
            # Diagnostics
            tactile_this_ep.append(obs_dict['tactile'])
            pc_this_ep.append(obs_dict['point_cloud'])
            # Reward breakdown (if available)
            if hasattr(env.envs[0], '_compute_reward'):
                r = env.envs[0]._compute_reward()
                rewards_this_ep.append(r)
                # Chamfer and occupancy
                if hasattr(env.envs[0], 'point_count') and env.envs[0].point_count > 10:
                    chamfer = env.envs[0]._compute_chamfer_distance()
                    chamfers_this_ep.append(chamfer)
                occ_this_ep.append(len(env.envs[0].voxel_grid))
            # Point cloud corruption for sanity check
            if corrupt_pc and hasattr(env.envs[0], 'point_cloud'):
                env.envs[0].point_cloud += np.random.normal(0, 0.1, env.envs[0].point_cloud.shape)
            if done:
                break
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tactile_readings.append(np.array(tactile_this_ep))
        point_clouds.append(np.array(pc_this_ep))
        reward_breakdown.append(np.array(rewards_this_ep))
        chamfer_deltas.append(np.array(chamfers_this_ep))
        occupancy_gains.append(np.array(occ_this_ep))
        action_stats.append(np.array(actions_this_ep))
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    env.close()
    # Generate plots
    plot_results(episode_rewards, episode_lengths, tactile_readings, point_clouds, reward_breakdown, chamfer_deltas, occupancy_gains, action_stats)

def heuristic_policy(obs):
    # Example: move in a random direction, or grid-based exploration
    return np.random.uniform(-1, 1, size=(2,)).astype(np.float32)

def plot_results(rewards, lengths, tactile_readings, point_clouds, reward_breakdown, chamfer_deltas, occupancy_gains, action_stats):
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, 'b-', label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/rewards.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, 'r-', label='Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/lengths.png')
    plt.close()
    # Tactile readings (last episode)
    if tactile_readings:
        plt.figure(figsize=(12, 6))
        tactile_data = np.array(tactile_readings[-1])
        plt.imshow(tactile_data.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Sensor Reading')
        plt.xlabel('Time Step')
        plt.ylabel('Sensor ID')
        plt.title('Tactile Sensor Readings (Last Episode)')
        plt.savefig('figures/tactile_readings.png')
        plt.close()
    # Point cloud (last frame)
    if point_clouds:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        points = point_clouds[-1][-1]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud (Last Frame)')
        plt.savefig('figures/point_cloud.png')
        plt.close()
    # Reward breakdown
    if reward_breakdown:
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(reward_breakdown[-1]), label='Reward (last ep)')
        plt.title('Reward Breakdown (Last Episode)')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('figures/reward_breakdown.png')
        plt.close()
    # Chamfer deltas
    if chamfer_deltas and len(chamfer_deltas[-1]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(chamfer_deltas[-1], label='Chamfer Delta (last ep)')
        plt.title('Chamfer Distance Evolution (Last Episode)')
        plt.xlabel('Step')
        plt.ylabel('Chamfer Delta')
        plt.legend()
        plt.savefig('figures/chamfer_deltas.png')
        plt.close()
    # Occupancy gains
    if occupancy_gains:
        plt.figure(figsize=(10, 6))
        plt.plot(occupancy_gains[-1], label='Voxel Occupancy (last ep)')
        plt.title('Voxel Occupancy Evolution (Last Episode)')
        plt.xlabel('Step')
        plt.ylabel('Voxel Count')
        plt.legend()
        plt.savefig('figures/occupancy_gains.png')
        plt.close()
    # Action stats
    if action_stats:
        plt.figure(figsize=(10, 6))
        actions = np.array(action_stats[-1])
        plt.plot(actions)
        plt.title('Action Evolution (Last Episode)')
        plt.xlabel('Step')
        plt.ylabel('Action Value')
        plt.savefig('figures/action_stats.png')
        plt.close()

if __name__ == "__main__":
    model_path = "models/saved/final_model.zip"
    config_path = "src/configs/training_config.yaml"
    visualize_model(model_path, config_path, num_episodes=2, max_steps=300, epsilon_greedy=0.1, use_heuristic=False, corrupt_pc=False, object_model=None) 