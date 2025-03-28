"""
ContactProcessor Module

This module defines a class that processes raw contact sensor readings
and maps them into 3D space based on sensor positions in the scene.
"""

import torch
import math

class ContactProcessor:
    def __init__(self, sensor_names, scene, device="cpu"):
        """
        Initializes the ContactProcessor.

        Args:
            sensor_names (list[str]): List of contact sensor keys in the scene.
            scene: The current simulation scene object.
            device (str): Device to move tensors to (e.g., "cuda" or "cpu").
        """
        self.scene = scene
        self.sensor_names = sensor_names
        self.device = device
        self.history = []  # Stores historical contact positions and forces

    def generate_hemisphere_sensor_poses(self, radius=0.02, center_offset=0.065, num_points=64):
        """
        Generate sensor positions in the robot local frame (half sphere).

        Returns:
            torch.Tensor: Tensor of shape [num_points, 3] in base frame
        """
        golden_ratio = (1 + 5 ** 0.5) / 2
        indices = torch.arange(0, num_points)
        theta = 2 * math.pi * indices / golden_ratio
        phi = torch.acos(1 - indices / num_points)
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi) + center_offset
        return torch.stack([x, y, z], dim=-1)  # [num_sensors, 3]

    def _get_sensor_positions(self):
        """
        Computes global positions of each contact sensor based on robot base pose.

        Returns:
            torch.Tensor: Tensor of shape [num_envs, num_sensors, 3]
        """
        local_pos = self.generate_hemisphere_sensor_poses(num_points=len(self.sensor_names)).to(self.device)
        base_pose = self.scene["robot"].data.root_state_w[:, :7]  # [num_envs, 7] (pos + quat)
        pos = base_pose[:, :3]  # [num_envs, 3]
        quat = base_pose[:, 3:7]  # [num_envs, 4]
        local_pos = local_pos.unsqueeze(0).expand(pos.shape[0], -1, -1)  # [num_envs, num_sensors, 3]
        global_pos = self._quat_apply(quat, local_pos) + pos.unsqueeze(1)
        return global_pos  # [num_envs, num_sensors, 3]

    def _quat_apply(self, quat, vec):
        """
        Applies quaternion rotation to vectors.
        Args:
            quat: [N, 4]
            vec: [N, M, 3]
        Returns:
            Rotated vectors: [N, M, 3]
        """
        qvec = quat[:, None, 1:]
        uv = torch.cross(qvec, vec, dim=-1)
        uuv = torch.cross(qvec, uv, dim=-1)
        return vec + 2 * (quat[:, None, :1] * uv + uuv)

    def get_contact_forces(self):
        """
        Retrieve net contact forces for each sensor.

        Returns:
            torch.Tensor: Force vectors of shape [num_envs, num_sensors, 3]
        """
        forces = []
        for name in self.sensor_names:
            f = self.scene[name].data.net_forces_w  # [num_envs, 3] or [num_envs, 1, 3]
            if f.ndim == 3:
                f = f.squeeze(1)
            forces.append(f)
        return torch.stack(forces, dim=1).to(self.device)

    def get_force_magnitudes(self):
        """
        Compute L2 norm of contact forces.

        Returns:
            torch.Tensor: Magnitudes of shape [num_envs, num_sensors]
        """
        forces = self.get_contact_forces()
        return torch.norm(forces, dim=-1)

    def get_force_vectors_in_space(self):
        """
        Get 3D positions and force vectors for visualization.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - positions: [num_envs, num_sensors, 3]
                - forces: [num_envs, num_sensors, 3]
        """
        positions = self._get_sensor_positions()  # refresh to match latest robot pose
        forces = self.get_contact_forces()
        self.history.append((positions.clone(), forces.clone()))  # Save history
        return positions, forces
