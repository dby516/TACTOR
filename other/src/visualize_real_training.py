import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Utility Functions ==========
def load_training_data(log_path):
    """Load training data from a .npz file."""
    data = np.load(log_path)
    return data

# ========== Visualization Functions ==========
def plot_training_curves(data, outdir):
    """Plot reward, contact ratio, and coverage curves."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(data['rewards'], label='Reward', color='tab:blue')
    plt.plot(data['contacts'], label='Contact Ratio', color='tab:orange')
    plt.plot(data['coverage'], label='Coverage', color='tab:green')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Curves')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'training_curves_real.png'))
    plt.close()

    # Moving average
    window = 10
    plt.figure(figsize=(12, 6))
    for key, color in zip(['rewards', 'contacts', 'coverage'], ['tab:blue', 'tab:orange', 'tab:green']):
        arr = data[key]
        ma = np.convolve(arr, np.ones(window)/window, mode='valid')
        plt.plot(ma, label=f'{key.capitalize()} (MA)', color=color)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.title('Moving Average of Training Curves')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'training_curves_moving_avg.png'))
    plt.close()

def plot_action_distribution(data, outdir):
    """Plot histogram and correlation of actions."""
    actions = data['actions']
    plt.figure(figsize=(8, 6))
    sns.histplot(actions.flatten(), bins=30, kde=True)
    plt.title('Action Value Distribution')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(outdir, 'action_distribution.png'))
    plt.close()

    # Action correlation
    plt.figure(figsize=(8, 6))
    corr = np.corrcoef(actions.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.title('Action Component Correlation')
    plt.xlabel('Action Component')
    plt.ylabel('Action Component')
    plt.savefig(os.path.join(outdir, 'action_correlation.png'))
    plt.close()

def plot_point_cloud_evolution(data, outdir):
    """Plot point cloud evolution at start, middle, and end."""
    pcs = data['point_clouds']
    fig = plt.figure(figsize=(15, 5))
    for i, idx in enumerate([0, len(pcs)//2, -1]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        pc = pcs[idx]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
        ax.set_title(f'Point Cloud at Step {idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'point_cloud_evolution_real.png'))
    plt.close()

def plot_tactile_heatmap(data, outdir):
    """Plot tactile sensor heatmap."""
    tactile = data['tactile']
    plt.figure(figsize=(10, 6))
    sns.heatmap(tactile.T, cmap='viridis')
    plt.title('Tactile Sensor Heatmap (Last Episode)')
    plt.xlabel('Step')
    plt.ylabel('Sensor')
    plt.savefig(os.path.join(outdir, 'tactile_heatmap_real.png'))
    plt.close()

# ========== Optional: Reward Breakdown, Chamfer, Occupancy ==========
def plot_optional_metrics(data, outdir):
    """Plot reward breakdown, Chamfer distance, occupancy gain if available."""
    if 'reward_breakdown' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['reward_breakdown'])
        plt.title('Reward Breakdown Over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward Component')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'reward_breakdown.png'))
        plt.close()
    if 'chamfer' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['chamfer'])
        plt.title('Chamfer Distance Over Time')
        plt.xlabel('Step')
        plt.ylabel('Chamfer Distance')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'chamfer_distance.png'))
        plt.close()
    if 'occupancy' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['occupancy'])
        plt.title('Occupancy Gain Over Time')
        plt.xlabel('Step')
        plt.ylabel('Voxel Count')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'occupancy_gain.png'))
        plt.close()

# ========== Main ==========
def main():
    log_path = 'logs/training_data_real.npz'
    outdir = 'paper-template-latex/images'
    data = load_training_data(log_path)
    plot_training_curves(data, outdir)
    plot_action_distribution(data, outdir)
    plot_point_cloud_evolution(data, outdir)
    plot_tactile_heatmap(data, outdir)
    plot_optional_metrics(data, outdir)
    print(f"Saved all real-data visualizations to {outdir}")

if __name__ == '__main__':
    main() 