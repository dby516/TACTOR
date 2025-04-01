import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform


from tactor_rl.tasks.tactor.tactor_rl_env_cfg import TactorEnvCfg
# from in_out.shape_proc.pointnet_feature_extraction import extract_pointnet_features
"""
 extract pointcloud feature:
 input:
        pointcloud: raw points (1, 3, N)
        model: pretrained PointNet model
    output:
        features
"""
class TacShapeExploreEnv(DirectRLEnv):
    cfg: TactorEnvCfg

    def __init__(self, cfg: TactorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.tactile_features = torch.zeros((self.num_envs, 64), device=self.device) # 64 sensors, 3-dim pressure vector

        # Accumulated contact points per env: list of [N_i, 3]
        self.pc_accum = [[] for _ in range(self.num_envs)]

        # Initialize with random points (e.g., in 10cm cube around contact area)
        for env_id in range(self.num_envs):
            rand_pts = 0.1 * (torch.rand((20, 3), device=self.device) - 0.5)
            self.pc_accum[env_id].extend(rand_pts.tolist())  # stored as list of [x, y, z]

        # PointNet model (pretrained)
        # self.pointnet_model = load_pointnet_model().to(self.device)
        # self.pointnet_model.eval()

        self.pointnet_feat = torch.zeros((self.num_envs, 256), device=self.device)  # example dim
        self.prev_pointnet_feat = self.pointnet_feat

    def _setup_scene(self):
        # Create robot and object
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot._has_implicit_actuators = False
        self.object = RigidObject(self.cfg.object_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # Add contact sensors
        self.contact_sensors = []
        for i, sensor_cfg in enumerate(self.cfg.contact_sensors):
            sensor = ContactSensor(cfg=sensor_cfg)
            self.contact_sensors.append(sensor)
            self.scene.sensors[f"contact_{i}"] = sensor  # register to scene

        # Add ground and lighting
        spawn_ground_plane("/World/defaultGroundPlane", GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        # for env_idx in range(self.num_envs):
        #     for sensor in self.contact_sensors:
        #         force = sensor.data.net_forces_w[env_idx, 0]  # shape: [3]
        #         if torch.norm(force) > 1e-5:
        #             pos = sensor.data.pos_w[env_idx, 0]  # shape: [3]
        #             self.pc_accum[env_idx].append(pos.tolist())

        # # Pad each environment’s point cloud
        # max_pts = max(len(pc) for pc in self.pc_accum)
        # pc_padded = torch.zeros((self.num_envs, 3, max_pts), device=self.device)

        # for i, pc in enumerate(self.pc_accum):
        #     pc_tensor = torch.tensor(pc, device=self.device).T  # [3, N]
        #     pc_padded[i, :, :pc_tensor.shape[1]] = pc_tensor

        # with torch.no_grad():
        #     self.pointnet_feat = extract_pointnet_features(pc_padded, self.pointnet_model)  # shape: [num_envs, 256]

        self.pointnet_feat = torch.rand((self.num_envs, 256), device=self.device)

    def _apply_action(self) -> None:
        """ Epsilon-greedy Exploration """
        # Parameters
        epsilon = 0.2  # exploration probability
        rand_mask = torch.rand((self.num_envs,), device=self.device) < epsilon

        # Random deltas: small translation noise
        delta_random = 0.01 * (2 * torch.rand_like(self.actions) - 1.0)

        # Blend actions: use random if mask is True, otherwise use policy action
        final_action = torch.where(rand_mask.unsqueeze(1), delta_random, self.actions)

        # Apply translation + keep fixed rotation
        root_pos = self.robot.data.root_pos_w.clone()
        root_quat = self.robot.data.root_quat_w.clone()
        new_pos = root_pos + final_action[:, :3]  # only translate

        new_pose = torch.cat([new_pos, root_quat], dim=-1)
        self.robot.write_root_pose_to_sim(new_pose)


    def _get_observations(self) -> dict:
        tactile_list = [sensor.data.net_forces_w.squeeze(1) for sensor in self.contact_sensors]
        self.tactile_feat = torch.cat(tactile_list, dim=-1)

        # Compute incremental of PointNet features
        pointnet_delta = torch.norm(self.pointnet_feat - self.prev_pointnet_feat, dim=-1, keepdim=True)
        self.prev_pointnet_feat = self.pointnet_feat.clone().detach()

        # === Actor input ===
        actor_obs = torch.cat([
            self.pointnet_feat,                  # [num_envs, 256]
            self.tactile_feat,                   # [num_envs, 192]
            self.robot.data.root_pos_w,          # [num_envs, 3]
            self.robot.data.root_quat_w,         # [num_envs, 4]
            self.actions                         # [num_envs, 6]
        ], dim=-1)

        # === Critic input ===
        critic_obs = torch.cat([
            self.pointnet_feat,
            pointnet_delta,
            self.tactile_feat
        ], dim=-1)

        return {"policy": actor_obs, "critic": critic_obs}



    def _get_rewards(self) -> torch.Tensor:
        feature_delta = self.tactile_features - torch.roll(self.tactile_features, 1, dims=0)
        reward = -torch.norm(feature_delta, dim=1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # ===== Reset objects or sensor states =====
        # Get the origin of each environment
        origins = self.scene.env_origins[env_ids]

        # Set local hovering position relative to the origin (e.g., z=0.6)
        local_hover = torch.tensor([0.0, 0.0, 0.8], device=self.device)
        pos = origins + local_hover  # [num_envs, 3]

        # Head-down orientation (180° around X-axis)
        axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        angle = torch.full((len(env_ids),), np.pi, device=self.device)
        quat = quat_from_angle_axis(angle, axis)  # [num_envs, 4]

        # Set robot pose
        pose = torch.cat([pos, quat], dim=-1)  # [num_envs, 7]
        self.robot.write_root_pose_to_sim(pose, env_ids)




@torch.jit.script
def quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(q1, quat_conjugate(q2))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))