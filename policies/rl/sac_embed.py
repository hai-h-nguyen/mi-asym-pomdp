import torch
from torch.optim import Adam
import numpy as np
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import torch.nn.functional as F


class SAC_Embed(RLAlgorithmBase):
    name = "sac-embed"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        init_temperature=1.0,
        automatic_entropy_tuning=True,
        target_entropy=None,
        temp_lr=3e-4,
        action_dim=None,
        **kwargs,
    ):

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
                self.target_entropy = -float(action_dim)
            self.log_alpha_entropy = torch.tensor(
                np.log(init_temperature), requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=temp_lr)
            self.alpha_lr = temp_lr
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        return actor(observ, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, _, _, log_probs = actor(observ, return_log_prob=True)
        return new_actions, log_probs  # (T+1, B, dim), (T+1, B, 1)

    def critic_loss(
        self,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from current policy,
            new_actions, new_log_probs = actor(
                observs=next_observs,
            )

            next_q1, next_q2 = critic_target(
                observs=next_observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)

            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q

        # Q(h(t), a(t)) (T, B, 1)
        q1_pred, q2_pred = critic(
            observs=observs,
            current_actions=actions,
        )  # (T, B, 1)
        return (q1_pred, q2_pred), q_target

    def actor_loss(
        self,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        expert_masks=None,
        rewards=None,
    ):
        new_actions, log_probs = actor(
            observs=observs
        )  # (T+1, B, A)

        q1, q2 = critic(
            observs=observs,
            current_actions=new_actions,
        )  # (T+1, B, 1)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        expert_loss = F.mse_loss(new_actions*expert_masks,
                                 actions*expert_masks)
        return policy_loss, log_probs, expert_loss
