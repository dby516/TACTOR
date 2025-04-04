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
        num_actor_obs=71,
        num_critic_obs=576,
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
        if activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        activation_fn = activation_map[activation.lower()]()

        self.pointnet_encoder = PointNetEncoderTrans(output_dim=256)

        # Contact processor for actor
        self.contact_branch = nn.Sequential(
            nn.Linear(64 * 3, 128),
            activation_fn,
            nn.Linear(128, 64),
            activation_fn
        )

        # Actor: contact + ee pose + pointnet feature
        self.actor_input_dim = 256 + 64 + 7
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, 256),
            activation_fn,
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, num_actions)
        )

        # Critic: difference of two point features + action
        self.critic_input_dim = 256 * 2 + 7
        self.critic = nn.Sequential(
            nn.Linear(self.critic_input_dim, 256),
            activation_fn,
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, 1)
        )

        # Noise
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
        pc_flat = actor_obs[:, start:start + 3 * N]               # [B, 3072]
        start += 3 * N
        tactile_flat = actor_obs[:, start:start + 64 * 3]         # [B, 192]
        start += 64 * 3
        pos_quat = actor_obs[:, start:start + 7]                  # [B, 7]

        pc = pc_flat.view(B, 3, N)                                # [B, 3, 1024]
        contact_encoded = self.contact_branch(tactile_flat)
        pointnet_feat = self.pointnet_encoder(pc)

        fused = torch.cat([pointnet_feat, contact_encoded, pos_quat], dim=-1)
        return fused

    def _split_critic_inputs(self, critic_obs):
        B = critic_obs.shape[0]
        num_points = 256
        pcd_dim = 3 * num_points  # 3072

        pcd_t_flat   = critic_obs[:, :pcd_dim]
        pcd_t_1_flat = critic_obs[:, pcd_dim:2*pcd_dim]
        actions      = critic_obs[:, 2*pcd_dim:]

        pcd_t   = pcd_t_flat.view(B, 3, num_points)
        pcd_t_1 = pcd_t_1_flat.view(B, 3, num_points)

        feat_t   = self.pointnet_encoder(pcd_t)
        feat_t_1 = self.pointnet_encoder(pcd_t_1)

        critic_input = torch.cat([feat_t, feat_t_1, actions], dim=-1)
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

