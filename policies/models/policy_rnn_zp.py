import torch
from copy import deepcopy
import torch.nn as nn
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic_zp import Critic_RNN
from policies.models.recurrent_actor_zp import Actor_RNN


class ModelFreeOffPolicy_Separate_RNN_ZP(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with separate RNNs
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        # pixel obs
        image_encoder_fn=lambda: None,
        image_size=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        # Critics
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            config_seq.model,
            config_rl.config_critic,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
            image_size=image_size,
        )
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=config_rl.critic_lr)
        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = Actor_RNN(
            obs_dim,
            action_dim,
            config_seq.model,
            config_rl.config_actor,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
        )
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=config_rl.actor_lr)
        # target networks
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self, *args, **kwargs):
        return self.actor.get_initial_info(*args, **kwargs)

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        state,
        deterministic=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)

        current_action, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
        )

        return current_action, current_internal_state

    def get_state_dict(self):
        dict = {}
        dict["actor"] = self.actor.state_dict()
        dict["actor_target"] = self.actor_target.state_dict()
        dict["critic"] = self.critic.state_dict()
        dict["critic_target"] = self.critic_target.state_dict()
        dict["actor_optimizer"] = self.actor_optimizer.state_dict()
        dict["critic_optimizer"] = self.critic_optimizer.state_dict()

        if 'sac' in self.algo.name:
            dict["sac"] = self.algo.get_special_dict()

        return dict

    def restore_state_dict(self, dict):
        self.actor.load_state_dict(dict["actor"])
        self.actor_target.load_state_dict(dict["actor_target"])
        self.critic.load_state_dict(dict["critic"])
        self.critic_target.load_state_dict(dict["critic_target"])
        self.actor_optimizer.load_state_dict(dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(dict["critic_optimizer"])

        if 'sac' in self.algo.name:
            self.algo.load_special_dict(dict["sac"])

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )

        outputs = {}
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. Critic loss
        (q1_pred, q2_pred), q_target, critic_info = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
        )

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        critic_pred = critic_info["pred_next_latent"] * masks
        critic_target = critic_info["target_next_latent"] * masks
        critic_zp_loss = ((critic_pred - critic_target)**2).sum() / num_valid
        critic_zp_loss *= critic_info["zp_loss_w"]
        qf1_loss += 0.5 * critic_zp_loss
        outputs.update({
            "critic_zp_loss": critic_zp_loss.item(),
        })

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()

        outputs.update({
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
        })

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad_norm
            )
            outputs["raw_critic_grad_norm"] = grad_norm.item()

        self.critic_optimizer.step()

        ### 2. Actor loss
        policy_loss, log_probs, actor_info = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
        )
        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid
        outputs["actor_loss"] = policy_loss.item()

        actor_pred = actor_info["pred_next_latent"] * masks
        actor_target = actor_info["target_next_latent"] * masks
        actor_zp_loss = ((actor_pred - actor_target)**2).sum() / num_valid
        actor_zp_loss *= actor_info["zp_loss_w"]
        policy_loss += actor_zp_loss
        outputs.update({
            "actor_zp_loss": actor_zp_loss.item(),
        })

        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_grad_norm
            )
            outputs["raw_actor_grad_norm"] = grad_norm.item()

        self.actor_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        ptu.soft_update_from_to(self.critic.seq_model,
                                self.critic.seq_model_target,
                                self.tau)

        ptu.soft_update_from_to(self.actor.seq_model,
                                self.actor.seq_model_target,
                                self.tau)

        ### 4. update others like alpha
        if log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "critic_grad_norm": utl.get_grad_norm(self.critic),
            "critic_seq_grad_norm": utl.get_grad_norm(self.critic.seq_model),
            "actor_grad_norm": utl.get_grad_norm(self.actor),
            "actor_seq_grad_norm": utl.get_grad_norm(self.actor.seq_model),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        batch_size = actions.shape[1]

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, states, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        outputs = self.forward(actions, rewards,
                               observs,
                               dones, masks)
        return outputs
