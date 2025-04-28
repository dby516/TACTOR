# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab.utils import configclass

from external_tools.point_net.models.pointnet_utils import PointNetEncoder


class PointNetEncoderTrans(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=False)
        self.projector = nn.Linear(1024, output_dim)

    def forward(self, x):
        feat_1024, _, _ = self.encoder(x)  # x: (B, 3, N)
        feat_256 = self.projector(feat_1024)
        return feat_256


@configclass
class TactorActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "tactor_rl.networks.tac_actor_critic.TactorActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [1024, 512, 256, 128]
    critic_hidden_dims: list[int] = [1024, 512, 256, 128]
    activation: str = "elu"


class TactorActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs=3463,  # 256*3 + 64*6 + 3 + 4 + 7
        num_critic_obs=3463, # same as actor
        num_actions=7,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="log",
        **kwargs,
    ):
        super().__init__()
        self.noise_std_type = noise_std_type
        self.action_dim = num_actions

        activation_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leakyrelu": nn.LeakyReLU,
        }
        activation_fn = activation_map.get(activation.lower(), nn.ELU)()

        self.pointnet_encoder = PointNetEncoderTrans(output_dim=256)

        # Contact processor for 64 sensors × (3 pos + 3 force)
        self.contact_branch = nn.Sequential(
            nn.Linear(64 * 6, 128),
            activation_fn,
            nn.Linear(128, 64),
            activation_fn
        )

        # Actor network: [PointNet + Contact + Pose]
        self.actor_input_dim = 256 + 64 + 7
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, actor_hidden_dims[0]),
            activation_fn,
            nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]),
            activation_fn,
            nn.Linear(actor_hidden_dims[1], num_actions)
        )

        # Critic network (same input for now)
        self.critic_input_dim = self.actor_input_dim + num_actions
        self.critic = nn.Sequential(
            nn.Linear(self.critic_input_dim, critic_hidden_dims[0]),
            activation_fn,
            nn.Linear(critic_hidden_dims[0], critic_hidden_dims[1]),
            activation_fn,
            nn.Linear(critic_hidden_dims[1], 1)
        )

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("Invalid noise_std_type")

        self.distribution = None

    def _split_actor_inputs(self, actor_obs):
        B = actor_obs.shape[0]
        N = 256

        start = 0
        pc_flat = actor_obs[:, start:start + 3 * N]
        start += 3 * N
        tactile_flat = actor_obs[:, start:start + 64 * 6]
        start += 64 * 6
        pos_quat = actor_obs[:, start:start + 7]

        pc = pc_flat.view(B, 3, N)
        contact_encoded = self.contact_branch(tactile_flat)
        pointnet_feat = self.pointnet_encoder(pc)

        fused = torch.cat([pointnet_feat, contact_encoded, pos_quat], dim=-1)
        return fused

    def _split_critic_inputs(self, critic_obs):
        B = critic_obs.shape[0]
        N = 256

        start = 0
        pc_flat = critic_obs[:, start:start + 3 * N]
        start += 3 * N
        tactile_flat = critic_obs[:, start:start + 64 * 6]
        start += 64 * 6
        pos_quat = critic_obs[:, start:start + 7]
        start += 7
        actions = critic_obs[:, start:start + self.action_dim]

        pc = pc_flat.view(B, 3, N)
        contact_encoded = self.contact_branch(tactile_flat)
        pointnet_feat = self.pointnet_encoder(pc)

        critic_input = torch.cat([pointnet_feat, contact_encoded, pos_quat, actions], dim=-1)
        return self.critic(critic_input)
    def compute_supervised_loss(self, pcd_accum, pcd_gt):
        feat_accum = self.pointnet_encoder(pcd_accum)  # [B, 256]
        feat_gt = self.pointnet_encoder(pcd_gt)        # [B, 256]
        loss = nn.functional.mse_loss(feat_accum, feat_gt.detach())
        return loss

    def update_distribution(self, observations):
        x = self._split_actor_inputs(observations)
        mean = self.actor(x)
        std = torch.exp(self.std) if self.noise_std_type == "log" else self.std
        std = std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        x = self._split_actor_inputs(observations)
        return self.actor(x)

    def evaluate(self, critic_observations, **kwargs):
        return self._split_critic_inputs(critic_observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

