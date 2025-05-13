import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VisualizationCallback(BaseCallback):
    """
    Custom callback for visualizing training progress in TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None
        
    def _on_training_start(self):
        """
        Initialize TensorBoard writer at the start of training.
        """
        self.writer = SummaryWriter(self.locals['tensorboard_log'])
        
    def _on_step(self):
        """
        Log custom metrics and visualizations at each step.
        """
        # Get current observation
        obs = self.locals['rollout_buffer'].observations
        
        # Visualize point cloud if available
        if 'point_cloud' in obs:
            point_cloud = obs['point_cloud'][0]  # Get first environment's point cloud
            self._visualize_point_cloud(point_cloud)
            
        # Visualize tactile sensor readings if available
        if 'tactile' in obs:
            tactile = obs['tactile'][0]  # Get first environment's tactile readings
            self._visualize_tactile_readings(tactile)
            
        # Log custom metrics
        self._log_custom_metrics()
        
        return True
        
    def _visualize_point_cloud(self, points):
        """
        Visualize point cloud data in TensorBoard.
        """
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')
        
        # Convert plot to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Add to TensorBoard
        self.writer.add_image('point_cloud', img, self.num_timesteps, dataformats='HWC')
        plt.close(fig)
        
    def _visualize_tactile_readings(self, tactile):
        """
        Visualize tactile sensor readings in TensorBoard.
        """
        # Create heatmap of tactile readings
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tactile.reshape(-1, 1), cmap='viridis')
        plt.colorbar(im)
        ax.set_title('Tactile Sensor Readings')
        
        # Convert plot to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Add to TensorBoard
        self.writer.add_image('tactile_readings', img, self.num_timesteps, dataformats='HWC')
        plt.close(fig)
        
    def _log_custom_metrics(self):
        """
        Log custom metrics to TensorBoard.
        """
        # Get current episode info
        infos = self.locals['infos']
        
        # Log episode length
        if 'episode' in infos[0]:
            episode_length = infos[0]['episode']['l']
            self.writer.add_scalar('episode/length', episode_length, self.num_timesteps)
            
        # Log episode reward
        if 'episode' in infos[0]:
            episode_reward = infos[0]['episode']['r']
            self.writer.add_scalar('episode/reward', episode_reward, self.num_timesteps)
            
        # Log action statistics
        actions = self.locals['rollout_buffer'].actions
        self.writer.add_histogram('actions', actions, self.num_timesteps)
        
    def _on_training_end(self):
        """
        Close TensorBoard writer at the end of training.
        """
        if self.writer is not None:
            self.writer.close() 