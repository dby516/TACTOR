# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


import torch
import torch.nn as nn
from torch.distributions import Normal

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab.utils import configclass

@configclass
class TactorActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "tactor_rl.networks.tac_actor_critic.TactorActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [1024, 512, 256, 128]
    critic_hidden_dims: list[int] = [1024, 512, 256, 128]
    activation: str = "elu"


class TactorActorCritic(nn.Module):
    is_recurrent = False  # required by RSL-RL

    def __init__(
        self,
        num_actor_obs=327,
        num_critic_obs=576,
        num_actions=7, # pos(3) + quat(4)
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

        # Shared contact encoder
        self.contact_branch = nn.Sequential(
            nn.Linear(64 * 3, 128),  # 64 contact sensors
            activation_fn,
            nn.Linear(128, 64),
            activation_fn
        )

        # === Actor ===
        self.actor_input_dim = 327  # pointnet + pose + contact encoded
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, 256),
            activation_fn,
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, num_actions)
        )

        # === Critic ===
        self.critic_input_dim = num_critic_obs # pointnet + infogain + contact encoded
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

    def _split_inputs(self, x):
        # Input order: [pointnet_feat(256), tactile_feat(192), ee_pos(3), ee_quat(4), actions(6)]
        pointnet_feat = x[:, :256]  # example dim
        contact_flat = x[:, 256:448]
        contact_encoded = self.contact_branch(contact_flat)
        ee_pose = x[:, 448:455] # ee_pos + ee_quat
        
        fused = torch.cat([pointnet_feat, ee_pose, contact_encoded], dim=-1)
        return fused

    def update_distribution(self, observations):
        x = self._split_inputs(observations)
        mean = self.actor(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        x = self._split_inputs(observations)
        return self.actor(x)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

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
