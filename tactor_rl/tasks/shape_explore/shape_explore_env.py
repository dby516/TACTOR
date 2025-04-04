# Updated TacShapeExploreEnv with torch.cat for flattened obs
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

class TacShapeExploreEnv(DirectRLEnv):
    cfg: TactorEnvCfg

    def __init__(self, cfg: TactorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.tactile_features = torch.zeros((self.num_envs, 64), device=self.device)  # 64 sensors * 3D

        self.max_points = 256
        self.pc_accum = [[] for _ in range(self.num_envs)]  # raw contact points
        self.pc_prev = torch.zeros((self.num_envs, 3, self.max_points), device=self.device)
        self.pc_padded = torch.zeros((self.num_envs, 3, self.max_points), device=self.device)

        for env_id in range(self.num_envs):
            rand_pts = 0.1 * (torch.rand((20, 3), device=self.device) - 0.5)
            self.pc_accum[env_id].extend(rand_pts.tolist())

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot._has_implicit_actuators = False
        self.object = RigidObject(self.cfg.object_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        self.contact_sensors = []
        for i, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(cfg=sensor_cfg)
            self.contact_sensors.append(sensor)
            self.scene.sensors[f"contact_{i}"] = sensor

        spawn_ground_plane("/World/defaultGroundPlane", GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        self.pc_prev.copy_(self.pc_padded)

        for env_idx in range(self.num_envs):
            for sensor in self.contact_sensors:
                if sensor.data.net_forces_w is None:
                    continue

                force = sensor.data.net_forces_w[env_idx, 0]
                if torch.norm(force) > 1e-5:
                    # pos = sensor.data.pos_w[env_idx, 0]
                    pos = self.robot.data.root_pos_w[env_idx, :]
                    self.pc_accum[env_idx].append(pos.tolist())
                    # print("pc accum: ", self.pc_accum[env_idx])

        for i, pc in enumerate(self.pc_accum):
            if len(pc) == 0:
                pc_tensor = torch.zeros((3, 1), device=self.device)
            else:
                pc_tensor = torch.as_tensor(pc, device=self.device, dtype=torch.float32).T
                pc_tensor = pc_tensor - pc_tensor.mean(dim=1, keepdim=True)

            n = min(pc_tensor.shape[1], self.max_points)
            self.pc_padded[i, :, :n] = 0
            self.pc_padded[i, :, :n] = pc_tensor[:, :n]

    def _apply_action(self) -> None:
        epsilon = 0.2
        nudge_steps = 10
        max_step = 0.01
        t = self.episode_length_buf
        device = self.device

        rand_mask = torch.rand((self.num_envs,), device=device) < epsilon
        delta_random = 0.01 * (2 * torch.rand_like(self.actions) - 1.0)

        downward_action = torch.zeros_like(self.actions)
        downward_action[:, 2] = -0.01
        nudge_mask = (t < nudge_steps)

        # Clamp main action before mixing
        clamped_action = self.actions.clone()
        clamped_action[:, :3] = torch.clamp(clamped_action[:, :3], -max_step, max_step)

        # Combine everything
        final_action = torch.where(rand_mask.unsqueeze(1), delta_random, clamped_action)
        final_action = torch.where(nudge_mask.unsqueeze(1), downward_action, final_action)

        # Apply
        root_pos = self.robot.data.root_pos_w.clone()
        root_quat = self.robot.data.root_quat_w.clone()
        new_pos = root_pos + final_action[:, :3]
        new_pose = torch.cat([new_pos, root_quat], dim=-1)
        self.robot.write_root_pose_to_sim(new_pose)



    def _get_observations(self) -> dict:
        B = self.num_envs

        tactile_list = [sensor.data.net_forces_w.squeeze(1) for sensor in self.contact_sensors]
        self.tactile_feat = torch.cat(tactile_list, dim=-1)  # [B, 64*3]

        actor_obs = torch.cat([
            self.pc_padded.view(B, -1),
            self.tactile_feat,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.actions
        ], dim=-1)  # shape [B, 206]

        critic_obs = torch.cat([
            self.pc_padded.view(B, -1),
            self.pc_prev.view(B, -1),
            self.actions
        ], dim=-1)  # shape [B, 3*256*2 + 7] = [B, 6143]

        return {
            "policy": actor_obs,
            "critic": critic_obs,
            "pointcloud": self.pc_padded  # if you still want to extract externally
        }

    def _get_rewards(self) -> torch.Tensor:
        device = self.device
        step = self.episode_length_buf  # [B]

        # Get total contact force magnitude per environment
        contact_magnitudes = torch.stack([
            sensor.data.net_forces_w.norm(dim=-1).squeeze(1).to(device)
            for sensor in self.contact_sensors
        ], dim=1)  # [B, num_sensors]

        # Max force per env (or you can use mean, sum, etc.)
        force = contact_magnitudes.max(dim=1).values  # [B]
        
        if force.sum() > 0:
            print("force: ", force.sum(dim=0))

        # Rescale force (optional): clip and normalize to [0, 1]
        force_clipped = torch.clamp(force, 0.0, 10.0)
        force_scaled = force_clipped / 10.0  # → in [0, 1]

        # Phase-based shaping
        bonus = torch.where(step < 50, 0.1, 0.0)                  # small exploration bonus
        penalty = torch.where(force_scaled == 0, -0.5, 0.0)       # punish no-contact after warmup

        reward = force_scaled + bonus + penalty  # final reward
        return reward



    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        origins = self.scene.env_origins[env_ids]
        local_hover = torch.tensor([0.0, 0.0, 0.8], device=self.device)
        pos = origins + local_hover

        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        angle = torch.full((len(env_ids),), np.pi, device=self.device)
        quat = quat_from_angle_axis(angle, axis)

        pose = torch.cat([pos, quat], dim=-1)
        self.robot.write_root_pose_to_sim(pose, env_ids)

        for env_id in env_ids:
            self.pc_accum[env_id] = []

@torch.jit.script
def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
