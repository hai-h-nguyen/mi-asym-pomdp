"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from policies.models.critic_on_off import Critic
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_DQN_MLP_On_Off(nn.Module):
    """
    standard off-policy Markovian Policy using MLP
    including DQN
    NOTE: it can only solve MDP problem, not POMDPs
    , it uses embedders for observations and actions
    """

    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        state_dim,
        config_seq,
        config_rl,
        config_repr,
        gamma=0.99,
        tau=5e-3,
        image_encoder_fn=lambda: None,
        state_embedder_dir=None,
        obs_embedder_dir=None,
        image_size=None,
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
        # Critics
        self.critic = Critic(
            obs_dim,
            action_dim,
            state_dim,
            config_seq.model,
            config_rl.config_critic,
            config_repr,
            self.algo,
            state_embedder_dir,
            obs_embedder_dir,
            image_encoder=image_encoder_fn(),  # separate weight
            image_size=image_size,
        )

        self.critic_optim = Adam(self.critic.parameters(),
                                 lr=config_rl.critic_lr)
        # target networks
        self.critic_target = copy.deepcopy(self.critic)

    @torch.no_grad()
    def act(self, obs, state, deterministic=True):
        return self.critic.act(obs, state, deterministic)

    def update(self, batch):
        observs, next_observs = batch["obs"], batch["obs2"]  # (B, dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]  # (B, dim)

        states, next_states = batch["state"], batch["state2"]  # (B, dim)

        ### 1. Critic loss
        q_pred, q_target = self.algo.critic_loss(
            markov_critic=self.Markov_Critic,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_observs=next_observs,
            next_states=next_states,
        )

        qf_loss = F.mse_loss(q_pred, q_target)  # TD error

        # update q networks
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        outputs = {}
        outputs.update({
            "critic_loss": qf_loss.item(),
            "q": q_pred.mean().item(),
        })

        # soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
