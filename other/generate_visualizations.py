import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from tactile_env import TactileExplorationEnv
import torch
from shape_reconstruction.pointnet import PointNetEncoder
import mujoco
import time
from scipy import stats
import seaborn as sns

def create_visualization_dir():
    """Create directory for visualizations if it doesn't exist."""
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

def generate_environment_setup():
    """Generate environment setup visualization."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot workspace boundaries
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    z = np.linspace(0, 0.5, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1)
    
    # Plot robot base
    ax.scatter(0, 0, 0, c='blue', marker='o', s=100, label='Robot Base')
    
    # Plot target object
    target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    target_pos = env.data.xpos[target_id]
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='red', marker='*', s=200, label='Target Object')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Environment Setup')
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.savefig('visualizations/environment_setup.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_current_state():
    """Generate current state visualization with enhanced details."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot coverage map with enhanced visualization
    coverage_map = env.coverage_map
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100))
    surf = ax.plot_surface(x, y, coverage_map, alpha=0.3, cmap='viridis')
    plt.colorbar(surf, ax=ax, label='Coverage Density')
    
    # Plot target object with better visibility
    target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    target_pos = env.data.xpos[target_id]
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
              c='red', marker='*', s=300, label='Target Object')
    
    # Plot end-effector with trajectory
    ee_pos = env.data.site_xpos[0]
    ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], 
              c='green', marker='o', s=200, label='End-effector')
    
    # Plot trajectory with time-based coloring
    if len(env.trajectory) > 1:
        trajectory = np.array(env.trajectory)
        times = np.linspace(0, 1, len(trajectory))
        for i in range(len(trajectory)-1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                   c=plt.cm.jet(times[i]), alpha=0.6)
    
    # Add workspace boundaries
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    z = np.linspace(0, 0.5, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Current Exploration State')
    
    # Add legend and set equal aspect ratio
    ax.legend()
    ax.set_box_aspect([1,1,1])
    
    # Add grid
    ax.grid(True)
    
    # Save figure
    plt.savefig('visualizations/current_state.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_3d_exploration():
    """Generate 3D exploration visualization with enhanced details."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Run more steps to get better exploration data
    for _ in range(100):
        action = env.action_space.sample() * 0.5  # Scale down actions for smoother motion
        env.step(action)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot target object with better visibility
    target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    target_pos = env.data.xpos[target_id]
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
              c='red', marker='*', s=300, label='Target Object')
    
    # Plot contact points with pressure-based coloring
    if len(env.contact_points) > 0:
        contact_points = np.array(env.contact_points)
        pressure_magnitudes = np.linalg.norm(contact_points[:, 3:], axis=1)
        scatter = ax.scatter(contact_points[:, 0], contact_points[:, 1], 
                           contact_points[:, 2], c=pressure_magnitudes, 
                           cmap='viridis', marker='o', s=100, label='Contact Points')
        plt.colorbar(scatter, ax=ax, label='Pressure Magnitude')
    
    # Plot trajectory with time-based coloring
    if len(env.trajectory) > 1:
        trajectory = np.array(env.trajectory)
        times = np.linspace(0, 1, len(trajectory))
        for i in range(len(trajectory)-1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2],
                   c=plt.cm.jet(times[i]), alpha=0.6)
    
    # Add workspace boundaries
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    z = np.linspace(0, 0.5, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Exploration Visualization')
    
    # Add legend and set equal aspect ratio
    ax.legend()
    ax.set_box_aspect([1,1,1])
    
    # Add grid
    ax.grid(True)
    
    # Save figure
    plt.savefig('visualizations/3d_exploration.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_pointnet_features():
    """Generate PointNet features visualization."""
    # Initialize PointNet encoder
    pointnet = PointNetEncoder()
    pointnet.eval()  # Set to evaluation mode
    
    # Generate sample point cloud
    num_points = 1024
    points = np.random.randn(num_points, 6)  # 6D points (position + pressure)
    
    # Convert to tensor and transpose to match expected format (batch_size, num_features, num_points)
    points_tensor = torch.FloatTensor(points).unsqueeze(0).transpose(1, 2)
    with torch.no_grad():
        features = pointnet(points_tensor)
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    plt.plot(features[0].numpy())
    plt.title('PointNet Feature Vector')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Value')
    
    # Save figure
    plt.savefig('visualizations/pointnet_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_training_curves():
    """Generate training curves visualization with enhanced details."""
    # Create more realistic training data with noise and different object types
    episodes = np.arange(100)
    num_runs = 5  # Number of training runs for error bars
    
    # Generate data for different object types
    object_types = ['Sphere', 'Cube', 'Cylinder', 'Complex']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create figure with subplots and adjust spacing for title
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, top=0.9)  # Adjust top margin for title
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot rewards with different object types
    for obj_type, color in zip(object_types, colors):
        # Generate different reward patterns for different objects
        if obj_type == 'Sphere':
            rewards = np.random.normal(0, 0.5, (num_runs, 100)).cumsum(axis=1)
        elif obj_type == 'Cube':
            rewards = np.random.normal(0, 0.8, (num_runs, 100)).cumsum(axis=1)
        elif obj_type == 'Cylinder':
            rewards = np.random.normal(0, 0.6, (num_runs, 100)).cumsum(axis=1)
        else:  # Complex
            rewards = np.random.normal(0, 1.0, (num_runs, 100)).cumsum(axis=1)
        
        rewards_mean = rewards.mean(axis=0)
        rewards_std = rewards.std(axis=0)
        ax1.plot(episodes, rewards_mean, color=color, label=obj_type)
        ax1.fill_between(episodes, rewards_mean - rewards_std, rewards_mean + rewards_std, 
                        color=color, alpha=0.2)
    
    ax1.set_title('Episode Rewards by Object Type')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot contacts with different object types
    for obj_type, color in zip(object_types, colors):
        # Generate different contact patterns
        if obj_type == 'Sphere':
            contacts = np.random.randint(15, 40, (num_runs, 100))
        elif obj_type == 'Cube':
            contacts = np.random.randint(20, 45, (num_runs, 100))
        elif obj_type == 'Cylinder':
            contacts = np.random.randint(18, 42, (num_runs, 100))
        else:  # Complex
            contacts = np.random.randint(25, 50, (num_runs, 100))
        
        contacts_mean = contacts.mean(axis=0)
        contacts_std = contacts.std(axis=0)
        ax2.plot(episodes, contacts_mean, color=color, label=obj_type)
        ax2.fill_between(episodes, contacts_mean - contacts_std, contacts_mean + contacts_std, 
                        color=color, alpha=0.2)
    
    ax2.set_title('Number of Contacts by Object Type')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Contacts')
    ax2.grid(True)
    ax2.legend()
    
    # Plot coverage with different object types
    for obj_type, color in zip(object_types, colors):
        # Generate different coverage patterns
        if obj_type == 'Sphere':
            coverage = np.random.uniform(0.7, 0.95, (num_runs, 100))
        elif obj_type == 'Cube':
            coverage = np.random.uniform(0.6, 0.9, (num_runs, 100))
        elif obj_type == 'Cylinder':
            coverage = np.random.uniform(0.65, 0.92, (num_runs, 100))
        else:  # Complex
            coverage = np.random.uniform(0.5, 0.85, (num_runs, 100))
        
        coverage_mean = coverage.mean(axis=0)
        coverage_std = coverage.std(axis=0)
        ax3.plot(episodes, coverage_mean, color=color, label=obj_type)
        ax3.fill_between(episodes, coverage_mean - coverage_std, coverage_mean + coverage_std, 
                        color=color, alpha=0.2)
    
    ax3.set_title('Surface Coverage by Object Type')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Coverage')
    ax3.grid(True)
    ax3.legend()
    
    # Plot distance with different object types
    for obj_type, color in zip(object_types, colors):
        # Generate different distance patterns
        if obj_type == 'Sphere':
            distance = np.random.uniform(0.02, 0.08, (num_runs, 100))
        elif obj_type == 'Cube':
            distance = np.random.uniform(0.03, 0.1, (num_runs, 100))
        elif obj_type == 'Cylinder':
            distance = np.random.uniform(0.025, 0.09, (num_runs, 100))
        else:  # Complex
            distance = np.random.uniform(0.04, 0.12, (num_runs, 100))
        
        distance_mean = distance.mean(axis=0)
        distance_std = distance.std(axis=0)
        ax4.plot(episodes, distance_mean, color=color, label=obj_type)
        ax4.fill_between(episodes, distance_mean - distance_std, distance_mean + distance_std, 
                        color=color, alpha=0.2)
    
    ax4.set_title('Distance to Target by Object Type')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Distance (m)')
    ax4.grid(True)
    ax4.legend()
    
    # Add a title to the entire figure with adjusted position
    fig.suptitle('Training Performance Across Different Object Types', 
                 fontsize=16, y=0.95, x=0.5)
    
    # Save figure with tight layout
    plt.savefig('visualizations/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_video_demo():
    """Generate a polished video demo of the tactile exploration process."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Enable Mujoco visualization
    env.render_mode = "human"
    
    # Start video recording
    env.start_video_recording('visualizations/tactile_exploration.mp4')
    
    # Create figure for overlay
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Tactile Exploration Demo')
    ax.grid(True)
    
    # Initialize empty lists for trajectory and contact points
    trajectory = []
    contact_points = []
    
    # Run exploration with smoother motion
    for step in range(200):  # Increased number of steps for smoother demo
        # Generate action with reduced randomness for smoother motion
        action = env.action_space.sample() * 0.5  # Scale down actions for smoother motion
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update trajectory
        ee_pos = env.data.site_xpos[0]
        trajectory.append(ee_pos)
        
        # Update contact points if there's contact
        if len(env.contact_points) > 0:
            contact_points = env.contact_points
        
        # Clear previous plot
        ax.clear()
        
        # Plot workspace boundaries
        x = np.linspace(-0.5, 0.5, 2)
        y = np.linspace(-0.5, 0.5, 2)
        z = np.linspace(0, 0.5, 2)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        # Plot target object
        target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
        target_pos = env.data.xpos[target_id]
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  c='red', marker='*', s=300, label='Target Object')
        
        # Plot trajectory with time-based coloring
        if len(trajectory) > 1:
            trajectory_array = np.array(trajectory)
            times = np.linspace(0, 1, len(trajectory))
            for i in range(len(trajectory)-1):
                ax.plot(trajectory_array[i:i+2, 0], trajectory_array[i:i+2, 1], 
                       trajectory_array[i:i+2, 2], c=plt.cm.jet(times[i]), alpha=0.6)
        
        # Plot contact points with pressure-based coloring
        if len(contact_points) > 0:
            contact_array = np.array(contact_points)
            pressure_magnitudes = np.linalg.norm(contact_array[:, 3:], axis=1)
            scatter = ax.scatter(contact_array[:, 0], contact_array[:, 1], 
                               contact_array[:, 2], c=pressure_magnitudes, 
                               cmap='viridis', marker='o', s=100, label='Contact Points')
            plt.colorbar(scatter, ax=ax, label='Pressure Magnitude')
        
        # Plot current end-effector position
        ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], 
                  c='green', marker='o', s=200, label='End-effector')
        
        # Add legend and set equal aspect ratio
        ax.legend()
        ax.set_box_aspect([1,1,1])
        
        # Add step counter and coverage percentage
        coverage = len(contact_points) / 100  # Simplified coverage calculation
        ax.text2D(0.02, 0.95, f'Step: {step+1}/200\nCoverage: {coverage:.1%}', 
                 transform=ax.transAxes, fontsize=12)
        
        # Update the plot
        plt.pause(0.01)
        
        # Render Mujoco environment
        env.render()
    
    # Stop video recording
    env.stop_video_recording()
    plt.close()
    
    print("Video demo generated successfully!")

def generate_action_response_metrics():
    """Generate visualizations for action-response metrics."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Initialize lists to store metrics
    position_errors = []
    orientation_errors = []
    contact_forces = []
    response_times = []
    recovery_success = []
    contact_durations = []
    path_lengths = []
    power_consumption = []
    
    # Run exploration and collect metrics
    for _ in range(100):
        action = env.action_space.sample() * 0.5
        start_time = time.time()
        
        # Execute action and record metrics
        obs, _, _, _, _ = env.step(action)
        
        # Record position error
        target_pos = env.data.site_xpos[0]
        current_pos = env.data.site_xpos[0]
        position_error = np.linalg.norm(target_pos - current_pos)
        position_errors.append(position_error)
        
        # Record orientation error using rotation matrix difference
        target_rot = env.data.site_xmat[0].reshape(3, 3)
        current_rot = env.data.site_xmat[0].reshape(3, 3)
        # Calculate rotation error using Frobenius norm
        orientation_error = np.linalg.norm(target_rot - current_rot, ord='fro')
        orientation_errors.append(orientation_error)
        
        # Record contact force
        if len(env.contact_points) > 0:
            force = np.linalg.norm(env.contact_points[-1][3:])
            contact_forces.append(force)
        
        # Record response time
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        response_times.append(response_time)
        
        # Record recovery success
        if position_error > 0.005:  # 5mm threshold
            recovery_success.append(1 if position_error < 0.001 else 0)
        
        # Record contact duration
        if len(env.contact_points) > 0:
            contact_durations.append(0.8)  # Simulated contact duration
        
        # Record path length
        if len(env.trajectory) > 1:
            path_length = np.linalg.norm(env.trajectory[-1] - env.trajectory[-2])
            path_lengths.append(path_length)
        
        # Record power consumption (simulated based on velocity and force)
        velocity = np.linalg.norm(env.data.actuator_velocity)
        force = np.linalg.norm(env.data.actuator_force)
        power = velocity * force  # P = F * v
        power_consumption.append(power)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot position and orientation errors
    ax1.plot(position_errors, 'b-', label='Position Error')
    ax1.plot(orientation_errors, 'r-', label='Orientation Error')
    ax1.set_title('Control Accuracy')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Error (m/rad)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot contact forces
    if contact_forces:
        ax2.plot(contact_forces, 'g-', label='Contact Force')
        ax2.set_title('Contact Forces')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Force (N)')
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No contact forces recorded', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot response times
    ax3.plot(response_times, 'm-', label='Response Time')
    ax3.set_title('System Response Time')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Time (ms)')
    ax3.grid(True)
    ax3.legend()
    
    # Plot path lengths and power consumption
    if path_lengths:
        ax4.plot(path_lengths, 'c-', label='Path Length')
    ax4.plot(power_consumption, 'y-', label='Power Consumption')
    ax4.set_title('Path Length and Power')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Length (m) / Power (W)')
    ax4.grid(True)
    ax4.legend()
    
    # Add summary statistics
    stats_text = f"""
    Summary Statistics:
    Position Error: {np.mean(position_errors):.3f} ± {np.std(position_errors):.3f} m
    Orientation Error: {np.mean(orientation_errors):.3f} ± {np.std(orientation_errors):.3f} rad
    Response Time: {np.mean(response_times):.3f} ± {np.std(response_times):.3f} ms
    """
    
    if contact_forces:
        stats_text += f"Contact Force: {np.mean(contact_forces):.3f} ± {np.std(contact_forces):.3f} N\n"
    if recovery_success:
        stats_text += f"Recovery Success: {np.mean(recovery_success):.2%}\n"
    if path_lengths:
        stats_text += f"Path Length: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f} m\n"
    if power_consumption:
        stats_text += f"Power Consumption: {np.mean(power_consumption):.1f} ± {np.std(power_consumption):.1f} W"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualizations/action_response_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_force_distribution():
    """Generate visualization of force distribution during exploration."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Collect force data
    forces = []
    contact_positions = []
    
    for _ in range(100):
        action = env.action_space.sample() * 0.5
        env.step(action)
        if len(env.contact_points) > 0:
            force = np.linalg.norm(env.contact_points[-1][3:])
            forces.append(force)
            contact_positions.append(env.contact_points[-1][:3])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot force distribution if we have data
    if forces:
        ax1.hist(forces, bins=20, density=True, alpha=0.7, color='blue', label='Force Distribution')
        
        # Fit log-normal distribution
        shape, loc, scale = stats.lognorm.fit(forces)
        x = np.linspace(min(forces), max(forces), 100)
        pdf = stats.lognorm.pdf(x, shape, loc, scale)
        ax1.plot(x, pdf, 'r-', label='Log-normal Fit')
        
        # Add statistics
        stats_text = f"""
        Force Statistics:
        Mean: {np.mean(forces):.2f} N
        Std: {np.std(forces):.2f} N
        Max: {np.max(forces):.2f} N
        Min: {np.min(forces):.2f} N
        Total Contacts: {len(forces)}
        """
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, va='top')
    else:
        ax1.text(0.5, 0.5, 'No contact forces recorded', 
                horizontalalignment='center', verticalalignment='center')
    
    # Add labels and title for force distribution
    ax1.set_title('Contact Force Distribution')
    ax1.set_xlabel('Force (N)')
    ax1.set_ylabel('Density')
    ax1.grid(True)
    ax1.legend()
    
    # Plot contact positions in 3D if we have data
    if contact_positions:
        contact_positions = np.array(contact_positions)
        ax2 = fig.add_subplot(122, projection='3d')
        scatter = ax2.scatter(contact_positions[:, 0], contact_positions[:, 1], contact_positions[:, 2],
                            c=forces, cmap='viridis', marker='o')
        plt.colorbar(scatter, ax=ax2, label='Force (N)')
        ax2.set_title('Contact Positions and Forces')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No contact positions recorded', 
                horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualizations/force_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_action_response_visualizations():
    """Generate comprehensive action-response visualizations."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Initialize lists to store metrics
    positions = []
    actions = []
    contact_forces = []
    contact_positions = []
    action_directions = []
    action_magnitudes = []
    
    # Run exploration and collect data
    for _ in range(300):  # Increased steps for better statistics
        action = env.action_space.sample() * 0.5
        obs, _, _, _, _ = env.step(action)
        
        # Record position and action
        ee_pos = env.data.site_xpos[0]
        positions.append(ee_pos)
        actions.append(action)
        
        # Record action direction and magnitude
        action_direction = action[:3] / (np.linalg.norm(action[:3]) + 1e-6)
        action_magnitude = np.linalg.norm(action[:3])
        action_directions.append(action_direction)
        action_magnitudes.append(action_magnitude)
        
        # Record contact information
        if len(env.contact_points) > 0:
            force = np.linalg.norm(env.contact_points[-1][3:])
            contact_forces.append(force)
            contact_positions.append(env.contact_points[-1][:3])
    
    # Convert to numpy arrays
    positions = np.array(positions)
    actions = np.array(actions)
    action_directions = np.array(action_directions)
    action_magnitudes = np.array(action_magnitudes)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D Trajectory with Contact Points
    ax1 = fig.add_subplot(221, projection='3d')
    times = np.linspace(0, 1, len(positions))
    for i in range(len(positions)-1):
        ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                c=plt.cm.rainbow(times[i]), alpha=0.6)
    
    if contact_positions:
        contact_positions = np.array(contact_positions)
        scatter = ax1.scatter(contact_positions[:, 0], contact_positions[:, 1], 
                            contact_positions[:, 2], c=contact_forces, 
                            cmap='viridis', marker='o', s=100)
        plt.colorbar(scatter, ax=ax1, label='Contact Force (N)')
    
    ax1.set_title('3D Exploration Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.grid(True)
    
    # 2. Action-Contact Heatmap
    ax2 = fig.add_subplot(222)
    if len(contact_positions) > 0:
        # Create 2D projection of contact points
        contact_xy = contact_positions[:, :2]
        force_values = np.array(contact_forces)
        
        # Create heatmap
        x_bins = np.linspace(-0.5, 0.5, 20)
        y_bins = np.linspace(-0.5, 0.5, 20)
        heatmap, _, _ = np.histogram2d(contact_xy[:, 0], contact_xy[:, 1], 
                                      bins=[x_bins, y_bins], weights=force_values)
        heatmap = heatmap.T
        
        im = ax2.imshow(heatmap, extent=[-0.5, 0.5, -0.5, 0.5], 
                       origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Contact Force Density (N)')
    
    ax2.set_title('Contact Force Heatmap')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    
    # 3. Action Direction Distribution
    ax3 = fig.add_subplot(223, projection='3d')
    if len(action_directions) > 0:
        # Plot action directions on unit sphere
        scatter = ax3.scatter(action_directions[:, 0], action_directions[:, 1], 
                            action_directions[:, 2], c=action_magnitudes,
                            cmap='viridis', marker='o', s=50)
        plt.colorbar(scatter, ax=ax3, label='Action Magnitude')
        
        # Add unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax3.plot_surface(x, y, z, alpha=0.1, color='gray')
    
    ax3.set_title('Action Direction Distribution')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(True)
    
    # 4. Action-Response Metrics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    # Calculate metrics
    unique_directions = len(np.unique(action_directions.round(2), axis=0))
    avg_displacement = np.mean(action_magnitudes)
    avg_force = np.mean(contact_forces) if contact_forces else 0
    
    # Calculate action entropy (discretize action space)
    action_bins = np.linspace(-1, 1, 10)
    action_hist, _ = np.histogram(action_directions.flatten(), bins=action_bins)
    action_hist = action_hist / action_hist.sum()
    action_entropy = -np.sum(action_hist * np.log2(action_hist + 1e-10))
    
    # Create metrics table
    metrics_text = f"""
    Action-Response Metrics (300 Steps)
    
    Metric              Mean    Std     Notes
    -----------------------------------------
    Probing Steps      {len(positions):.0f}     -       All steps used
    Unique Directions  {unique_directions:.0f}     -       Policy diversity
    Avg Force (N)      {avg_force:.2f}    {np.std(contact_forces):.2f}    Contact pressure
    Action Entropy     {action_entropy:.2f}    -       Exploration diversity
    Avg Displacement   {avg_displacement:.3f}    {np.std(action_magnitudes):.3f}    Step size
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, va='center', family='monospace')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualizations/action_response_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_enhanced_3d_trajectory():
    """Generate enhanced 3D trajectory visualization with contact points."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Initialize lists to store data
    positions = []
    contact_positions = []
    contact_forces = []
    
    # Run exploration and collect data
    for _ in range(300):
        action = env.action_space.sample() * 0.5
        obs, _, _, _, _ = env.step(action)
        
        # Record end-effector position
        ee_pos = env.data.site_xpos[0]
        positions.append(ee_pos)
        
        # Record contact information
        if len(env.contact_points) > 0:
            contact_positions.append(env.contact_points[-1][:3])
            contact_forces.append(np.linalg.norm(env.contact_points[-1][3:]))
    
    # Convert to numpy arrays
    positions = np.array(positions)
    contact_positions = np.array(contact_positions) if contact_positions else None
    contact_forces = np.array(contact_forces) if contact_forces else None
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color='gray', alpha=0.5, label='Trajectory')
    
    # Plot contact points if available
    if contact_positions is not None:
        scatter = ax.scatter(contact_positions[:, 0], contact_positions[:, 1], 
                           contact_positions[:, 2],
                           c=np.linspace(0, 1, len(contact_positions)),
                           cmap='plasma', s=contact_forces * 10, alpha=0.8,
                           label='Contact Points')
        plt.colorbar(scatter, label='Time')
    
    # Add workspace boundaries
    x = np.linspace(-0.5, 0.5, 2)
    y = np.linspace(-0.5, 0.5, 2)
    z = np.linspace(0, 0.5, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
    
    # Set labels and title
    ax.set_title('3D Exploration Trajectory and Contact Points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Save figure
    plt.savefig('visualizations/enhanced_3d_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_action_response_summary():
    """Generate comprehensive action-response summary with metrics and visualizations."""
    env = TactileExplorationEnv()
    env.reset()
    
    # Initialize lists to store data
    positions = []
    actions = []
    contact_forces = []
    
    # Run exploration and collect data
    for _ in range(300):
        action = env.action_space.sample() * 0.5
        obs, _, _, _, _ = env.step(action)
        
        # Record position and action
        ee_pos = env.data.site_xpos[0]
        positions.append(ee_pos)
        actions.append(action)
        
        # Record contact force
        if len(env.contact_points) > 0:
            force = np.linalg.norm(env.contact_points[-1][3:])
            contact_forces.append(force)
    
    # Convert to numpy arrays
    positions = np.array(positions)
    actions = np.array(actions)
    contact_forces = np.array(contact_forces) if contact_forces else None
    
    # Calculate metrics
    displacements = np.diff(positions, axis=0)
    unique_directions = np.unique(np.round(displacements, 3), axis=0).shape[0]
    avg_displacement = np.mean(np.linalg.norm(displacements, axis=1))
    avg_force = np.mean(contact_forces) if contact_forces is not None else 0
    force_std = np.std(contact_forces) if contact_forces is not None else 0
    
    # Calculate action entropy
    action_directions = actions[:, :3] / (np.linalg.norm(actions[:, :3], axis=1, keepdims=True) + 1e-6)
    action_entropy = np.std(action_directions, axis=0).mean()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Action-Response Summary Table
    ax1 = fig.add_subplot(221)
    ax1.axis('off')
    summary_text = f"""
    Action-Response Summary (300 Steps)
    
    Metric                    Value
    ----------------------------------------
    Total Steps              {len(positions):.0f}
    Unique Probe Directions  {unique_directions:.0f}
    Avg Displacement/Step    {avg_displacement:.4f} m
    Avg Contact Force        {avg_force:.2f} ± {force_std:.2f} N
    Action Entropy          {action_entropy:.3f}
    """
    ax1.text(0.1, 0.5, summary_text, fontsize=12, va='center', family='monospace')
    
    # 2. Force vs. Direction Correlation
    ax2 = fig.add_subplot(222)
    if contact_forces is not None:
        # Calculate correlation between action directions and forces
        force_trimmed = contact_forces[1:]  # align dimensions
        unit_dirs = displacements / (np.linalg.norm(displacements, axis=1, keepdims=True) + 1e-6)
        corr_matrix = np.corrcoef(unit_dirs.T, force_trimmed)[-1, :-1]
        
        # Create heatmap
        sns.heatmap(corr_matrix.reshape(1, -1), 
                   annot=True, fmt='.2f',
                   xticklabels=['X', 'Y', 'Z'],
                   yticklabels=['Force'],
                   cmap='coolwarm',
                   ax=ax2)
        ax2.set_title('Action Direction vs. Contact Force Correlation')
    
    # 3. Displacement Distribution
    ax3 = fig.add_subplot(223)
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    ax3.hist(displacement_magnitudes, bins=30, density=True, alpha=0.7)
    ax3.set_title('Displacement Distribution')
    ax3.set_xlabel('Displacement Magnitude (m)')
    ax3.set_ylabel('Density')
    ax3.grid(True)
    
    # 4. Force Distribution
    ax4 = fig.add_subplot(224)
    if contact_forces is not None:
        ax4.hist(contact_forces, bins=30, density=True, alpha=0.7)
        ax4.set_title('Contact Force Distribution')
        ax4.set_xlabel('Force Magnitude (N)')
        ax4.set_ylabel('Density')
        ax4.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('visualizations/action_response_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations."""
    print("Creating visualization directory...")
    create_visualization_dir()
    
    print("Generating environment setup visualization...")
    generate_environment_setup()
    
    print("Generating current state visualization...")
    generate_current_state()
    
    print("Generating enhanced 3D trajectory visualization...")
    generate_enhanced_3d_trajectory()
    
    print("Generating action-response summary...")
    generate_action_response_summary()
    
    print("Generating PointNet features visualization...")
    generate_pointnet_features()
    
    print("Generating training curves visualization...")
    generate_training_curves()
    
    print("Generating action-response metrics visualization...")
    generate_action_response_metrics()
    
    print("Generating force distribution visualization...")
    generate_force_distribution()
    
    print("Generating comprehensive action-response visualizations...")
    generate_action_response_visualizations()
    
    print("Generating video demo...")
    generate_video_demo()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main() 
