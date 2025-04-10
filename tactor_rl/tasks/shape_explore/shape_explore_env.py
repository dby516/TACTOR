import torch 
import numpy as np
from collections.abc import Sequence
import matplotlib.pyplot as plt
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul

from tactor_rl.tasks.tactor.tactor_rl_env_cfg import TactorEnvCfg

class TacShapeExploreEnv(DirectRLEnv):
    cfg: TactorEnvCfg

    def __init__(self, cfg: TactorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.tactile_features = torch.zeros((self.num_envs, 64), device=self.device)

        self.max_points = 256
        self.pc_prev = torch.zeros((self.num_envs, 3, self.max_points), device=self.device)
        self.pc_padded = torch.zeros((self.num_envs, 3, self.max_points), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot._has_implicit_actuators = False
        self.object = RigidObject(self.cfg.object_cfg)
        self.has_joints = False
        
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        self.contact_sensors = []
        for i, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(cfg=sensor_cfg)
            self.contact_sensors.append(sensor)
            self.scene.sensors[f"contact_{i}"] = sensor

        # spawn_ground_plane("/World/defaultGroundPlane", GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.pc_prev.copy_(self.pc_padded)

        for i in range(self.num_envs):
            new_points = []
            for sensor in self.contact_sensors:
                if sensor.data.net_forces_w is None or sensor.data.pos_w is None:
                    continue
                force = sensor.data.net_forces_w[i, 0]
                if torch.norm(force) > 1e-5:
                    pos = sensor.data.pos_w[i, 0]
                    new_points.append(pos)

            if not new_points:
                continue

            new_points_tensor = torch.stack(new_points, dim=0)
            new_points_tensor -= new_points_tensor.mean(dim=0, keepdim=True)
            new_points_tensor = new_points_tensor.T
            print("gain points: ", len(new_points), "env: ", i)
            old_pc = self.pc_padded[i]
            valid_mask = torch.any(old_pc != 0, dim=0)
            old_valid = old_pc[:, valid_mask]

            total = old_valid.shape[1] + new_points_tensor.shape[1]
            if total <= self.max_points:
                updated = torch.cat([old_valid, new_points_tensor], dim=1)
            else:
                excess = total - self.max_points
                updated = torch.cat([old_valid[:, excess:], new_points_tensor], dim=1)

            self.pc_padded[i].zero_()
            self.pc_padded[i, :, :updated.shape[1]] = updated

            # if i == 0 and len(new_points) > 0:
            #     ep_len = self.episode_length_buf[i].item()
            #     pc = self.pc_padded[i, :, :updated.shape[1]].cpu().numpy()
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.scatter(pc[0], pc[1], pc[2], c='b', marker='o', s=2)
            #     ax.set_title("pc_padded[1] - Iteration")
            #     ax.set_xlim([-0.1, 0.1])
            #     ax.set_ylim([-0.1, 0.1])
            #     ax.set_zlim([-0.1, 0.1])

            #     os.makedirs("pc_plots", exist_ok=True)
            #     # plt.savefig(f"pc_plots/frame_{ep_len:04d}.png")
            #     plt.close()

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

        clamped_action = self.actions.clone()
        clamped_action[:, :] = torch.clamp(clamped_action[:, :], -max_step, max_step) #1, :3

        final_action = torch.where(rand_mask.unsqueeze(1), delta_random, clamped_action)
        final_action = torch.where(nudge_mask.unsqueeze(1), downward_action, final_action)

        root_pos = self.robot.data.root_pos_w.clone()
        root_quat = self.robot.data.root_quat_w.clone()
        new_pos = root_pos + final_action[:, :3]
        new_quat = root_quat + final_action[:, 3:]
        new_pose = torch.cat([new_pos, new_quat], dim=-1)
        old_pose = torch.cat([root_pos, root_quat], dim=-1) # debug
        self.robot.write_root_pose_to_sim(new_pose) # debug

    def _get_observations(self) -> dict:
        B = self.num_envs

        tactile_list = [sensor.data.net_forces_w.squeeze(1) for sensor in self.contact_sensors]
        self.tactile_feat = torch.cat(tactile_list, dim=-1)

        actor_obs = torch.cat([
            self.pc_padded.view(B, -1),
            self.tactile_feat,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.actions
        ], dim=-1)

        critic_obs = torch.cat([
            self.pc_padded.view(B, -1),
            self.pc_prev.view(B, -1),
            self.actions
        ], dim=-1)

        return {
            "policy": actor_obs,
            "critic": critic_obs,
            "pointcloud": self.pc_padded
        }

    # def _get_rewards(self) -> torch.Tensor:
    #     device = self.device
    #     step = self.episode_length_buf

    #     contact_magnitudes = torch.stack([
    #         sensor.data.net_forces_w.norm(dim=-1).squeeze(1).to(device)
    #         for sensor in self.contact_sensors
    #     ], dim=1)
    #     force = contact_magnitudes.max(dim=1).values
    #     force_scaled = torch.clamp(force, 0.0, 10.0) / 10.0

    #     duplicate_penalty = torch.zeros(self.num_envs, device=device)
    #     for env_idx in range(self.num_envs):
    #         pc_all = self.pc_padded[env_idx]
    #         pc_all = pc_all[:, torch.any(pc_all != 0, dim=0)]
    #         if pc_all.shape[1] == 0:
    #             continue

    #         pc_all_np = np.round(pc_all.T.cpu().numpy(), decimals=4)
    #         existing_set = set(map(tuple, pc_all_np))

    #         new_points = []
    #         for sensor in self.contact_sensors:
    #             if sensor.data.net_forces_w is None or sensor.data.pos_w is None:
    #                 continue
    #             force = sensor.data.net_forces_w[env_idx, 0]
    #             if torch.norm(force) > 1e-5:
    #                 pos = sensor.data.pos_w[env_idx, 0]
    #                 new_points.append(pos.cpu().numpy())

    #         if not new_points:
    #             continue

    #         new_points_np = np.round(np.stack(new_points), decimals=4)
    #         new_dup = sum(tuple(p) in existing_set for p in new_points_np)
    #         duplicate_penalty[env_idx] = 1.0 if new_dup >= 8 else 0.0

    #     bonus = torch.where(step < 50, 0.1, 0.0)
    #     penalty = torch.where(force_scaled == 0, -0.5, 0.0)

    #     reward = force_scaled + bonus + penalty - 0.5 * duplicate_penalty
    #     return reward
    def _get_rewards(self) -> torch.Tensor:
        step = self.episode_length_buf

        # Contact magnitude [B, S] where S = number of sensors
        contact_magnitudes = torch.stack([
            sensor.data.net_forces_w.norm(dim=-1).squeeze(1).to(self.device)
            for sensor in self.contact_sensors
        ], dim=1)  # [B, S]

        # Max force from any sensor
        max_force = contact_magnitudes.max(dim=1).values
        force_reward = torch.clamp(max_force, 0.0, 5.0) / 5.0  # scaled to [0, 1]

        # Compute contact novelty penalty (duplicate points)
        novelty_penalty = torch.zeros(self.num_envs, device=self.device)
        for env_idx in range(self.num_envs):
            pc_all = self.pc_padded[env_idx]
            pc_all = pc_all[:, torch.any(pc_all != 0, dim=0)]
            if pc_all.shape[1] == 0:
                continue

            existing_set = set(map(tuple, np.round(pc_all.T.cpu().numpy(), decimals=4)))
            new_points = [
                sensor.data.pos_w[env_idx, 0].cpu().numpy()
                for sensor in self.contact_sensors
                if sensor.data.net_forces_w is not None and torch.norm(sensor.data.net_forces_w[env_idx, 0]) > 1e-5
            ]

            if new_points:
                new_points_np = np.round(np.stack(new_points), decimals=4)
                num_duplicates = sum(tuple(p) in existing_set for p in new_points_np)
                novelty_penalty[env_idx] = num_duplicates / len(new_points_np)  # ratio of repeats

        # Encourage contact
        base_reward = force_reward

        # Discourage repeat contact
        base_reward -= 0.5 * novelty_penalty

        # Slight exploration incentive early on
        early_bonus = torch.where(step < 30, 0.05, 0.0)

        reward = base_reward + early_bonus
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

        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        angle = torch.full((len(env_ids),), -np.pi / 2, device=self.device)
        quat = quat_from_angle_axis(angle, axis)

        pose = torch.cat([pos, quat], dim=-1)

        if not self.has_joints:
            stage = self.scene.stage
            for i in range(self.num_envs):
                obj_path = f"/World/envs/env_{i}/Object"
                env_path = f"/World/envs/env_{i}"
                joint_name = f"fixedJoint_env_{i}"
                env_origin = self.scene.env_origins[i]
                x, y, z = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
                sim_utils.add_fixed_joint_to_world(stage, env_path, obj_path, (x, y, z), joint_name)
            self.has_joints = True


        self.robot.write_root_pose_to_sim(pose, env_ids)

@torch.jit.script
def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))