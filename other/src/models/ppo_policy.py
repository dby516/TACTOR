import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from .pointnet import PointNet

class TactileFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=128):
        super().__init__()
        # PointNet for point cloud processing
        self.pointnet = PointNet(feature_transform=True)
        # Tactile feature processing
        self.tactile_encoder = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Robot pose and action processing
        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # Combined feature processing
        combined_dim = 128 + 128 + 32 + 32  # pointnet + tactile + pose + action
        self.combined_encoder = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, obs):
        batch_size = obs['tactile'].size(0)
        # Process point cloud
        point_cloud = obs['point_cloud'].float()
        point_features, _, _ = self.pointnet(point_cloud)
        # Process tactile readings
        tactile = obs['tactile'].float()
        tactile_features = self.tactile_encoder(tactile)
        # Process robot pose
        pose = obs['robot_pose'].float()
        pose_features = self.pose_encoder(pose)
        # Process previous action
        prev_action = obs['prev_action'].float()
        action_features = self.action_encoder(prev_action)
        # Combine features
        combined = torch.cat([
            point_features,
            tactile_features,
            pose_features,
            action_features
        ], dim=1)
        features = self.combined_encoder(combined)
        return features

# Optionally, keep a stub for PPOPolicy for compatibility
class PPOPolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # PointNet for point cloud processing
        self.pointnet = PointNet(feature_transform=True)
        
        # Tactile feature processing
        self.tactile_encoder = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Robot pose and action processing
        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_dim = 128 + 128 + 32 + 32  # pointnet + tactile + pose + action
        self.combined_encoder = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6-DoF action
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        self.features_dim = 128  # Combined feature dimension after combined_encoder
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Dictionary containing:
                - tactile: Tactile sensor readings (B, 64, 6)
                - point_cloud: Accumulated point cloud (B, N, 3)
                - robot_pose: Robot pose (B, 7)
                - prev_action: Previous action (B, 6)
                
        Returns:
            action_mean: Mean of action distribution
            action_std: Standard deviation of action distribution
            value: Value estimate
        """
        batch_size = obs['tactile'].size(0)
        
        # Process point cloud
        point_cloud = obs['point_cloud'].float()
        point_features, _, _ = self.pointnet(point_cloud)
        
        # Process tactile readings
        tactile = obs['tactile'].float()
        tactile_features = self.tactile_encoder(tactile)
        
        # Process robot pose
        pose = obs['robot_pose'].float()
        pose_features = self.pose_encoder(pose)
        
        # Process previous action
        prev_action = obs['prev_action'].float()
        action_features = self.action_encoder(prev_action)
        
        # Combine features
        combined = torch.cat([
            point_features,
            tactile_features,
            pose_features,
            action_features
        ], dim=1)
        
        features = self.combined_encoder(combined)
        
        # Get action distribution parameters
        action_mean = self.policy_head(features)
        action_std = torch.ones_like(action_mean) * 0.1  # Fixed std for simplicity
        
        # Get value estimate
        value = self.value_head(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from observation.
        
        Args:
            obs: Observation dictionary
            deterministic: Whether to sample deterministically
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.size(0), 1)
        else:
            # Sample from normal distribution
            normal = torch.distributions.Normal(action_mean, action_std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observation dictionary
            action: Actions to evaluate
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
            value: Value estimate
        """
        action_mean, action_std, value = self.forward(obs)
        
        # Compute log probability and entropy
        normal = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal.log_prob(action).sum(dim=1, keepdim=True)
        entropy = normal.entropy().mean()
        
        return log_prob, entropy, value 