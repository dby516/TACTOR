import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from typing import Dict, Tuple, Optional
import torch

class TactileEnv(gym.Env):
    """Tactile exploration environment using MuJoCo."""
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 30,
    }
    
    def __init__(self, config: Dict, render_mode: Optional[str] = None):
        """
        Initialize the tactile exploration environment.
        
        Args:
            config: Configuration dictionary containing environment parameters
            render_mode: Optional render mode for visualization
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config['model_path'])
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_steps = config.get('max_steps', 300)
        self.current_step = 0
        self.voxel_size = config.get('voxel_size', 0.01)
        self.num_tactile_sensors = config.get('num_tactile_sensors', 64)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space includes:
        # - Tactile readings (len(self.data.sensordata))
        # - Accumulated point cloud (Nx3)
        # - Robot pose (7)
        # - Previous action (2)
        self.observation_space = spaces.Dict({
            'tactile': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.model.nsensor,), 
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(config.get('max_points', 1024), 3),
                dtype=np.float32
            ),
            'robot_pose': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),  # [x, y, z, qw, qx, qy, qz]
                dtype=np.float32
            ),
            'prev_action': spaces.Box(
                low=-1.0, high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        })
        
        # Initialize point cloud storage
        self.point_cloud = np.zeros((config.get('max_points', 1024), 3))
        self.point_count = 0
        
        # Initialize voxel grid for occupancy tracking
        self.voxel_grid = {}
        
        # Initialize renderer if needed
        if self.render_mode is not None:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
            )
        else:
            self.viewer = None
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset environment state
        self.current_step = 0
        self.point_cloud.fill(0)
        self.point_count = 0
        self.voxel_grid.clear()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 6-dimensional action vector [dx, dy, dz, droll, dpitch, dyaw]
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Apply action to robot
        self._apply_action(action)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Update point cloud with new contacts
        self._update_point_cloud()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to robot control."""
        # Scale action to appropriate ranges
        pos_scale = 0.01  # 1cm per step
        # Only use the available actuators
        self.data.ctrl[:] = action[:] * pos_scale
    
    def _update_point_cloud(self):
        """Update point cloud with new contact points."""
        # Get contact points from tactile sensors
        contacts = self._get_tactile_contacts()
        
        # Add new points to point cloud
        for contact in contacts:
            if self.point_count < self.point_cloud.shape[0]:
                self.point_cloud[self.point_count] = contact
                self.point_count += 1
                
                # Update voxel grid
                voxel_idx = self._point_to_voxel(contact)
                self.voxel_grid[tuple(voxel_idx)] = True
    
    def _get_tactile_contacts(self) -> np.ndarray:
        """Get contact points from tactile sensors."""
        contacts = []
        num_sensors = len(self.data.sensordata)
        for i in range(num_sensors):
            if self.data.sensordata[i] > 0:  # If sensor is in contact
                # Get contact position in world frame
                pos = self.data.sensordata[i:i+3]
                contacts.append(pos)
        return np.array(contacts)
    
    def _point_to_voxel(self, point: np.ndarray) -> np.ndarray:
        """Convert point to voxel index."""
        return np.floor(point / self.voxel_size).astype(int)
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        # Contact reward
        contact_reward = np.sum(self.data.sensordata[:self.num_tactile_sensors] > 0)
        
        # Voxel occupancy reward
        new_voxels = len(self.voxel_grid)
        voxel_reward = new_voxels * 0.1
        
        # Chamfer distance reward (if we have enough points)
        chamfer_reward = 0.0
        if self.point_count > 10:
            chamfer_reward = self._compute_chamfer_distance()
        
        # Combine rewards
        total_reward = (
            0.4 * contact_reward +
            0.3 * voxel_reward +
            0.3 * chamfer_reward
        )
        
        return total_reward
    
    def _compute_chamfer_distance(self) -> float:
        """Compute Chamfer distance between current and previous point clouds."""
        # This is a simplified version - in practice, you'd want to use
        # a proper Chamfer distance implementation
        if self.point_count < 2:
            return 0.0
            
        current_points = self.point_cloud[:self.point_count]
        prev_points = self.point_cloud[:self.point_count-1]
        
        # Compute pairwise distances
        diff = current_points[None, :, :] - prev_points[:, None, :]
        dist = np.sum(diff * diff, axis=2)
        
        # Get minimum distances
        min_dist_1 = np.min(dist, axis=0)
        min_dist_2 = np.min(dist, axis=1)
        
        return -(np.mean(min_dist_1) + np.mean(min_dist_2))
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        # Log the sensordata for debugging
        print(f"sensordata (len={len(self.data.sensordata)}): {self.data.sensordata}")
        return {
            'tactile': np.array(self.data.sensordata, dtype=np.float32),
            'point_cloud': self.point_cloud,
            'robot_pose': np.concatenate([
                self.data.qpos[:3],  # Position
                self.data.qpos[3:7]  # Quaternion
            ]),
            'prev_action': self.data.ctrl
        }
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'step': self.current_step,
            'point_count': self.point_count,
            'voxel_count': len(self.voxel_grid)
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
            )
            
        if self.render_mode == "human":
            self.viewer.sync()
            return None
        elif self.render_mode == "rgb_array":
            return self.viewer.render()
        elif self.render_mode == "depth_array":
            return self.viewer.render(depth=True)
            
    def close(self):
        """Close the environment and renderer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None 