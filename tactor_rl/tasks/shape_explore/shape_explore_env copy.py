import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform

import torch.nn as nn

from tactor_rl.tasks.tactor.tactor_rl_env_cfg import TactorEnvCfg

# class TacShapeExploreEnv(DirectRLEnv):
#     cfg: TactorEnvCfg

#     def __init__(self, cfg: TactorEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

#         self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
#         self.tactile_features = torch.zeros((self.num_envs, 64), device=self.device)  # 64 sensors * 3D

#         self.pc_accum = [[] for _ in range(self.num_envs)]  # raw contact points
#         self.pc_padded = torch.zeros((self.num_envs, 3, 1), device=self.device)


#         # Initialize with some random contacts
#         for env_id in range(self.num_envs):
#             rand_pts = 0.1 * (torch.rand((20, 3), device=self.device) - 0.5)
#             self.pc_accum[env_id].extend(rand_pts.tolist())

#     def _setup_scene(self):
#         self.robot = Articulation(self.cfg.robot_cfg)
#         self.robot._has_implicit_actuators = False
#         self.object = RigidObject(self.cfg.object_cfg)

#         self.scene.articulations["robot"] = self.robot
#         self.scene.rigid_objects["object"] = self.object

#         self.contact_sensors = []
#         for i, sensor_cfg in enumerate(self.cfg.contact_sensors):
#             sensor = ContactSensor(cfg=sensor_cfg)
#             self.contact_sensors.append(sensor)
#             self.scene.sensors[f"contact_{i}"] = sensor

#         spawn_ground_plane("/World/defaultGroundPlane", GroundPlaneCfg())
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#         self.scene.clone_environments(copy_from_source=False)

#     def _pre_physics_step(self, actions: torch.Tensor) -> None:
#         self.actions = actions.clone()

#         max_points = 512
#         for env_idx in range(self.num_envs):
#             for sensor in self.contact_sensors:
#                 force = sensor.data.net_forces_w[env_idx, 0]
#                 if torch.norm(force) > 1e-5:
#                     pos = sensor.data.pos_w[env_idx, 0]
#                     self.pc_accum[env_idx].append(pos.tolist())
#                     if len(self.pc_accum[env_idx]) > max_points:
#                         self.pc_accum[env_idx] = self.pc_accum[env_idx][-max_points:]

#         # Pad and normalize
#         max_pts = max(len(pc) for pc in self.pc_accum)
#         self.pc_padded = torch.zeros((self.num_envs, 3, max_pts), device=self.device)

#         for i, pc in enumerate(self.pc_accum):
#             if len(pc) == 0:
#                 pc_tensor = torch.zeros((3, 1), device=self.device)
#             else:
#                 pc_tensor = torch.as_tensor(pc, device=self.device, dtype=torch.float32).T
#                 pc_tensor = pc_tensor - pc_tensor.mean(dim=1, keepdim=True)
#             self.pc_padded[i, :, :pc_tensor.shape[1]] = pc_tensor

#     def _apply_action(self) -> None:
#         epsilon = 0.2
#         rand_mask = torch.rand((self.num_envs,), device=self.device) < epsilon
#         delta_random = 0.01 * (2 * torch.rand_like(self.actions) - 1.0)
#         final_action = torch.where(rand_mask.unsqueeze(1), delta_random, self.actions)

#         root_pos = self.robot.data.root_pos_w.clone()
#         root_quat = self.robot.data.root_quat_w.clone()
#         new_pos = root_pos + final_action[:, :3]
#         new_pose = torch.cat([new_pos, root_quat], dim=-1)
#         self.robot.write_root_pose_to_sim(new_pose)

#     def _get_observations(self) -> dict:
#         tactile_list = [sensor.data.net_forces_w.squeeze(1) for sensor in self.contact_sensors]
#         self.tactile_feat = torch.cat(tactile_list, dim=-1)

#         actor_obs = torch.cat([
#             self.tactile_feat,
#             self.robot.data.root_pos_w,
#             self.robot.data.root_quat_w,
#             self.actions
#         ], dim=-1)

#         critic_obs = torch.cat([
#             self.pc_padded,
#             self.tactile_feat,
#             self.robot.data.root_pos_w,
#             self.robot.data.root_quat_w,
#             self.actions
#         ], dim=-1)
        
#         return {
#             "policy": actor_obs,
#             "critic": critic_obs,
#             "pointcloud": self.pc_padded
#         }

#     def _get_rewards(self) -> torch.Tensor:
#         return torch.zeros(self.num_envs, device=self.device)

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         time_out = self.episode_length_buf >= self.max_episode_length - 1
#         return torch.zeros_like(time_out), time_out

#     def _reset_idx(self, env_ids: Sequence[int] | None):
#         if env_ids is None:
#             env_ids = self.robot._ALL_INDICES
#         super()._reset_idx(env_ids)

#         origins = self.scene.env_origins[env_ids]
#         local_hover = torch.tensor([0.0, 0.0, 0.8], device=self.device)
#         pos = origins + local_hover

#         axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
#         angle = torch.full((len(env_ids),), np.pi, device=self.device)
#         quat = quat_from_angle_axis(angle, axis)

#         pose = torch.cat([pos, quat], dim=-1)
#         self.robot.write_root_pose_to_sim(pose, env_ids)

#         for env_id in env_ids:
#             self.pc_accum[env_id] = []

# @torch.jit.script
# def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
#     quat_diff = quat_mul(q1, quat_conjugate(q2))
#     return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


# Updated TacShapeExploreEnv with accumulation and exposure of contact points

class TacShapeExploreEnv(DirectRLEnv):
    def __init__(self, cfg: TactorEnvCfg, spacing: float = 2.5, max_points: int = 1024):
        super().__init__()
        self.cfg = cfg
        self.spacing = spacing
        self.max_points = max_points
        self.num_envs = 6

        self.origins = self._define_origins(num_origins=self.num_envs, spacing=spacing)
        self._setup_scene()

        # Buffers to store accumulated contact points
        self.current_points = torch.randn(self.num_envs, max_points, 3) * 0.01
        self.previous_points = self.current_points.clone()
        self.point_counts = torch.zeros(self.num_envs, dtype=torch.long)  # how many real points stored so far

        # (optional) Placeholder for groundtruth pointclouds
        self.groundtruth_points = None  # [num_envs, N_gt, 3]

    def _define_origins(self, num_origins: int, spacing: float) -> list[list[float]]:
        env_origins = torch.zeros(num_origins, 3)
        num_rows = int(np.floor(np.sqrt(num_origins)))
        num_cols = int(np.ceil(num_origins / num_rows))
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
        env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
        env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
        env_origins[:, 2] = 0.0
        return env_origins.tolist()

    def _setup_scene(self):
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        for i, origin in enumerate(self.origins):
            ns = f"/World/envs/env_{i}"
            prim_utils.create_prim(ns, "Xform", translation=origin)

            table_pos = (origin[0], origin[1], origin[2] + 0.8)
            robot_pos = (origin[0], origin[1], origin[2] + 0.8)
            object_pos = (origin[0], origin[1] + 0.25, origin[2] + 0.85)

            table_cfg = self.cfg.table_cfg.spawn
            table_cfg.func(f"{ns}/Table", table_cfg, translation=table_pos)

            robot_cfg = self.cfg.robot_cfg.replace(prim_path=f"{ns}/Robot")
            robot_cfg.init_state.pos = robot_pos
            robot = Articulation(cfg=robot_cfg)
            self.articulations[f"ur10e_{i}"] = robot

            object_usd = self.cfg.object_usd_list[i]
            object_cfg = UsdFileCfg(usd_path=object_usd, scale=self.cfg.object_scale)
            object_cfg.func(f"{ns}/Object", object_cfg, translation=object_pos)

    def get_scene_entities(self):
        return self.articulations

    def get_origins(self):
        return self.origins

    def reset_accumulated_contacts(self):
        self.previous_points = self.current_points.clone()
        self.current_points = torch.randn(self.num_envs, self.max_points, 3) * 0.01
        self.point_counts.zero_()

    def append_contact_points(self, env_ids: torch.Tensor, new_points: torch.Tensor):
        # new_points: [E, K, 3] where K <= remaining slots
        for i, env_id in enumerate(env_ids):
            count = self.point_counts[env_id].item()
            k = new_points.shape[1]
            next_count = min(self.max_points, count + k)
            self.current_points[env_id, count:next_count] = new_points[i, :next_count - count]
            self.point_counts[env_id] = next_count

    def get_observations(self):
        """
        Returns:
            actor_obs: dict with keys ['pointcloud', 'policy']
            critic_obs: dict with keys ['pcd_t', 'pcd_t_1', 'action']
        """
        actor_obs = {
            "pointcloud": self.current_points.permute(0, 2, 1),  # [B, 3, N]
            "policy": ...  # fill in with raw contact + pose info
        }

        critic_obs = {
            "pcd_t": self.current_points.permute(0, 2, 1),
            "pcd_t_1": self.previous_points.permute(0, 2, 1),
            "action": ...  # fill in with previous actions
        }

        return actor_obs, critic_obs

    def set_groundtruth_pointclouds(self, gt_pcd: torch.Tensor):
        self.groundtruth_points = gt_pcd  # shape: [B, N_gt, 3]
