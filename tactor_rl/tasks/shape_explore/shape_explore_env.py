import torch 
import numpy as np
from collections.abc import Sequence
import matplotlib.pyplot as plt
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.sensors import ContactSensor
from isaaclab.scene import InteractiveScene
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul

from tactor_rl.tasks.tactor.tactor_rl_env_cfg import TactorEnvCfg
from tactor_rl.tasks.shape_explore.reward_funcs import Reward_ChamferDist, Reward_Voxel_Occupancy, farthest_point_sampling

class TacShapeExploreEnv(DirectRLEnv):
    cfg: TactorEnvCfg

    def __init__(self, cfg: TactorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.tactile_features = torch.zeros((self.num_envs, 64), device=self.device)
        self.pointnet_encoder = None  # to be injected later

        self.max_points = 256
        self.pc_prev = self.init_pc()
        self.pc_padded = self.pc_prev.clone()

    def init_pc(self):
        # Create random points inside a circle in XY, fixed Z = 0.25
        num_envs = self.num_envs
        num_points = self.max_points
        device = self.device

        # Sample radius ∈ [0, 0.1] with sqrt to preserve uniform density
        r = 0.1 * torch.sqrt(torch.rand((num_envs, num_points), device=device))
        theta = 2 * torch.pi * torch.rand((num_envs, num_points), device=device)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.full_like(x, 0.25)

        return torch.stack([x, y, z], dim=1)

    def _setup_scene(self):
        # self.scene = InteractiveScene(self.cfg.scene)
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]

        self.contact_sensors = []
        for i in range(64):
            sensor = self.scene[f"contact_sensor_{i}"]
            self.contact_sensors.append(sensor)

        spawn_ground_plane("/World/defaultGroundPlane", GroundPlaneCfg(), translation=(0, 0, -50.0))
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.pc_prev = self.pc_padded.clone()

        for i in range(self.num_envs):
            new_points = []
            detected_sensors = []
            for idx, sensor in enumerate(self.contact_sensors):
                if sensor.data.net_forces_w is None or sensor.data.pos_w is None:
                    continue
                force = sensor.data.net_forces_w[i, 0]
                if torch.norm(force) > 1e-5 and torch.norm(force) < 1e3:
                    pos = sensor.data.pos_w[i, 0]
                    pos = pos - self.scene.env_origins[i]
                    new_points.append(pos)
                    detected_sensors.append(idx)

            if not new_points:
                continue
            # print(f"Gain {len(detected_sensors)} new points in env{i}: {detected_sensors}")

            new_points_tensor = torch.stack(new_points, dim=0)  # [N_new, 3]

            # Get valid points from current pc
            old_pc = self.pc_padded[i]
            valid_mask = torch.any(old_pc != 0, dim=0)
            old_valid = old_pc[:, valid_mask].T  # [N_old, 3]

            # Combine new and old points
            all_points = torch.cat([old_valid, new_points_tensor], dim=0)  # [N, 3]

            # Farthest point sampling
            sampled_pc = farthest_point_sampling(all_points, self.max_points)  # [k, 3]
            sampled_pc = sampled_pc.T  # [3, k]

            # Update pc_padded
            self.pc_padded[i].zero_()
            self.pc_padded[i, :, :self.max_points] = sampled_pc

            # Optional visualization
            if len(new_points) > 0:
                ep_len = self.episode_length_buf[i].item()
                pc = self.pc_padded[i].cpu().numpy()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pc[0], pc[1], pc[2], c='b', marker='o', s=2)
                ax.set_title(f"env_{i} step {ep_len}")
                ax.set_xlim([-0.3, 0.3])
                ax.set_ylim([-0.3, 0.3])
                ax.set_zlim([-0.3, 0.3])
                os.makedirs("pc_plots", exist_ok=True)
                plt.savefig(f"pc_plots/env_{i}.png")
                plt.close()


    def _apply_action(self) -> None:
        epsilon = 0.2                     # probability of taking a random action
        nudge_steps = 4                 # number of steps to nudge downward
        nudge_speed = 0.01              # Z-axis downward movement
        max_step = 0.02
        action_noise_std = 0.0
        device = self.device

        # Initialize the persistent step counter if not already
        if not hasattr(self, "nudge_step_counter"):
            self.nudge_step_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

        # Create action masks
        rand_mask = torch.rand((self.num_envs,), device=device) < epsilon
        delta_random = 0.01 * (2 * torch.rand_like(self.actions) - 1.0)

        # Fixed downward movement (only on Z)
        toward_center_action = torch.zeros_like(self.actions)
        toward_center_action[:, 2] = - nudge_speed

        # Determine if we are still in nudge phase
        nudge_mask = self.nudge_step_counter < nudge_steps

        # Increment the step counter
        self.nudge_step_counter += 1

        # Clamp the raw actions
        noisy_actions = self.actions + action_noise_std * torch.randn_like(self.actions)
        clamped_action = torch.clamp(noisy_actions, -max_step, max_step)

        # Choose between random and clamped actions
        final_action = torch.where(rand_mask.unsqueeze(1), delta_random, clamped_action)

        # Override with downward nudge for early steps
        final_action = torch.where(nudge_mask.unsqueeze(1), toward_center_action, final_action)

        # Apply action to update root pose
        root_pos = self.robot.data.root_pos_w.clone()
        root_quat = self.robot.data.root_quat_w.clone()
        new_pos = root_pos + final_action[:, :3]
        new_quat = root_quat + final_action[:, 3:]
        new_pose = torch.cat([new_pos, new_quat], dim=-1)
        old_pose = torch.cat([root_pos, root_quat], dim=-1)

        self.robot.write_root_pose_to_sim(new_pose)

    def _get_observations(self) -> dict:
        B = self.num_envs

        # Stack net forces and positions across sensors → [B, S, 6]
        forces = [sensor.data.net_forces_w.squeeze(1) for sensor in self.contact_sensors]  # each [B, 3]
        positions = [sensor.data.pos_w.squeeze(1) for sensor in self.contact_sensors]      # each [B, 3]

        # Concatenate [force | pos] → [B, S, 6]
        tactile_list = [torch.cat([f, p], dim=-1) for f, p in zip(forces, positions)]
        self.tactile_feat = torch.stack(tactile_list, dim=1).view(B, -1)  # [B, S * 6]

        actor_obs = torch.cat([
            self.pc_padded.view(B, -1),
            self.tactile_feat,
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.actions
        ], dim=-1)

        critic_obs = torch.cat([
            self.pc_padded.view(B, -1), # [B, 3*max_points(256)]
            self.tactile_feat, # [B, S*(3+3)]
            self.robot.data.root_pos_w, # [B, 3]
            self.robot.data.root_quat_w, # [B, 4]
            self.actions # [B, 7]
        ], dim=-1)

        return {
            "policy": actor_obs,
            "critic": critic_obs,
            "pointcloud": self.pc_padded
        }

    
    def _get_rewards(self) -> torch.Tensor:
        step = self.episode_length_buf  # [B]
        B = self.num_envs
        S = len(self.contact_sensors)

        # Contact magnitudes [B, S]
        contact_magnitudes = torch.stack([
            sensor.data.net_forces_w.norm(dim=-1).squeeze(1).to(self.device)
            for sensor in self.contact_sensors
        ], dim=1)

        avg_force = contact_magnitudes.mean(dim=1)  # [B]
        contact_detected = ((avg_force > 1e-3) & (avg_force < 1e2)).float()

        # Normalized contact count
        contact_len = (contact_magnitudes > 1e-3).sum(dim=1).float() / S  # [B]

        # Decay force reward over time
        decay = 1.0 - step / self.max_episode_length  # [B]
        force_weight = torch.clamp(decay, 0.0, 1.0)   # [B]

        # Force reward
        force_raw = (
            0.2 * contact_detected * torch.clamp(avg_force, 0.0, 5.0) / 5.0 +
            0.8 * contact_len
        )
        force_reward = force_weight * force_raw

        # Point clouds
        pc = self.pc_padded.clone().transpose(1, 2)       # [B, N, 3]
        pc_prev = self.pc_prev.clone().transpose(1, 2)    # [B, N, 3]

        voxel_rewards = torch.zeros(B, device=self.device)
        chamfer_rewards = torch.zeros(B, device=self.device)

        for i in range(B):
            c_len = int((contact_len[i] * S).item())

            if torch.allclose(pc[i], pc_prev[i], atol=1e-6):
                chamfer = 0.0
            else:
                raw_chamfer = Reward_ChamferDist(c_len, pc[i], pc_prev[i])
                raw_chamfer = min(raw_chamfer, 0.1)
                chamfer = raw_chamfer / 0.1 - 0.2
                # print(f"[Chamfer] Env {i}: raw={raw_chamfer:.4f}, scaled={chamfer:.4f}")

            voxel = Reward_Voxel_Occupancy(c_len, pc[i], pc_prev[i]) * 5.0

            voxel_rewards[i] = torch.clamp(torch.tensor(voxel, device=self.device), -0.2, 0.8)
            chamfer_rewards[i] = torch.tensor(chamfer, device=self.device)

        # Early contact bonus
        early_bonus = ((step < 30).float() * contact_detected) * 0.1

        # Distance penalty
        root_pos = self.robot.data.root_pos_w  # [B, 3]
        local_pos = root_pos - self.scene.env_origins  # [B, 3]
        dist_from_center = torch.norm(local_pos, dim=1)  # [B]
        # distance_penalty = torch.clamp((dist_from_center - 0.54), min=0.0) * -1.0  # negative reward if > 0.54

        # Final reward
        reward = (
            25 * force_reward +
            2 * chamfer_rewards +
            25 * voxel_rewards +
            0.001 * distance_penalty +
            early_bonus
        )
        # if reward.mean().item() is not 0:
        #     print("[DEBUG] Rewards -> force: {:.4f}, chamfer: {:.4f}, voxel: {:.4f}, distance_penalty: {:.4f}, early_bonus: {:.4f}".format(
        #         force_reward.mean().item()*20,
        #         chamfer_rewards.mean().item()*2,
        #         voxel_rewards.mean().item()*25,
        #         distance_penalty.mean().item()*0.1,
        #         early_bonus.mean().item() if isinstance(early_bonus, torch.Tensor) else early_bonus,
        #     ))

        return reward




    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
    #     return torch.zeros_like(time_out), time_out
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, 3] root positions in world frame
        root_pos = self.robot.data.root_pos_w  # [B, 3]
        local_pos = root_pos - self.scene.env_origins  # [B, 3]
        
        # Distance from object center
        dist_from_center = torch.norm(local_pos, dim=1)  # [B]

        # Termination: end if distance > 2.0
        too_far = dist_from_center > 2.0  # [B]

        # Timeout if max episode length reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return too_far, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot position
        origins = self.scene.env_origins[env_ids]
        local_hover = torch.tensor([0.0, 0.0, 0.38], device=self.device)
        pos = origins + local_hover

        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        angle = torch.full((len(env_ids),), -np.pi / 2, device=self.device)
        quat = quat_from_angle_axis(angle, axis)

        pose = torch.cat([pos, quat], dim=-1)
        self.robot.write_root_pose_to_sim(pose, env_ids)

        # Reset helping arrays
        self.pc_prev[env_ids] = 0.0
        self.pc_padded[env_ids] = 0.0

        

@torch.jit.script
def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))