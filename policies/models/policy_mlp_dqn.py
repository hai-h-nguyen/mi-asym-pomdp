"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from policies.models import *
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_DQN_MLP(nn.Module):
    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_rl,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )
        # Markov q networks
        self.qf = self.algo.build_critic(
            input_size=obs_dim,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
        )
        self.qf_optim = Adam(self.qf.parameters(), lr=lr)
        # target networks
        self.qf_target = copy.deepcopy(self.qf)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        return self.algo.select_action(
            qf=self.qf,
            observ=obs,
            deterministic=deterministic,
        )

    def update(self, batch):
        observs, next_observs = batch["obs"], batch["obs2"]  # (B, dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]  # (B, dim)

        outputs = {}

        ### 1. Critic loss
        q_pred, q_target, _ = self.algo.critic_loss(
            markov_critic=self.Markov_Critic,
            critic=self.qf,
            critic_target=self.qf_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            states=observs,
            next_observs=next_observs,
        )

        qf_loss = F.mse_loss(q_pred, q_target)  # TD error

        # update q networks
        self.qf_optim.zero_grad()
        qf_loss.backward()

        outputs.update({
            "critic_loss": qf_loss.item(),
            "q": q_pred.mean().item(),
        })

        self.qf_optim.step()

        # soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf, self.qf_target, self.tau)
