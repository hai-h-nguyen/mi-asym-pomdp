"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from policies.models.critic_embed import Critic_Embed as Critic
from policies.models.actor_embed import Actor_Embed as Actor
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_MLP_Embed(nn.Module):
    """
    standard off-policy Markovian Policy using MLP
    including TD3 and SAC
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
        config_seq,
        config_rl,
        gamma=0.99,
        tau=5e-3,
        image_encoder_fn=lambda: None,
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
            config_seq.model,
            config_rl.config_critic,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
            image_size=image_size,
        )

        self.critic_optim = Adam(self.critic.parameters(),
                                 lr=config_rl.critic_lr)
        # target networks
        self.critic_target = copy.deepcopy(self.critic)

        # Markov Actor
        self.actor = Actor(
            obs_dim,
            action_dim,
            config_seq.model,
            config_rl.config_actor,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
            image_size=image_size,
        )
        self.actor_optim = Adam(self.actor.parameters(),
                                lr=config_rl.actor_lr)
        # target network
        self.actor_target = copy.deepcopy(self.actor)

        # expert loss weight
        self.exp_w = config_rl.config_actor.expert_w

    @torch.no_grad()
    def act(self, obs, state, deterministic=False, return_log_prob=False):
        return self.actor.act(obs, deterministic)

    def update(self, batch):
        observs, next_observs = batch["obs"], batch["obs2"]  # (B, dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]  # (B, dim)
        expert_masks = batch["expert"]

        ### 1. Critic loss
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_observs=next_observs,
        )

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # update q networks
        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        # soft update
        self.soft_target_update()

        ### 2. Actor loss
        policy_loss, log_probs, expert_loss = self.algo.actor_loss(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            expert_masks=expert_masks,
        )
        policy_loss = policy_loss.mean()

        outputs = {
            "actor_loss": policy_loss.item(),
        }

        if self.exp_w > 0.0:
            policy_loss += expert_loss * self.exp_w
            outputs.update({"expert_loss": expert_loss.item()})

        # update policy network
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        outputs.update({
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": q1_pred.mean().item(),
            "q2": q2_pred.mean().item(),
        })

        # update others like alpha
        if log_probs is not None:
            current_log_probs = log_probs.mean().item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)
