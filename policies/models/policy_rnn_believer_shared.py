import torch
from copy import deepcopy
from torch.distributions.normal import Normal
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.seq_models import SEQ_MODELS
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.believer_encoders import BeliefVAEModel, RepresentationModel


class ModelFreeOffPolicy_Shared_RNN_Believer(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNNs
    """

    ARCH = "memory"

    def __init__(
        self,
        obs_dim,
        action_dim,
        env_name,
        config_seq,
        config_rl,
        freeze_critic: bool,
        # pixel obs
        image_encoder_fn=lambda: None,
        state_embedder_dir=None,
        obs_embedder_dir=None,
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

        self.freeze_critic = freeze_critic

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        config_seq = config_seq.model
        if image_encoder_fn() is None:
            observ_embedding_size = config_seq.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = deepcopy(image_encoder_fn())
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, config_seq.action_embedder.hidden_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(
            1, config_seq.reward_embedder.hidden_size, F.relu
        )

        ## 2. build RNN model
        rnn_input_size = (
            observ_embedding_size
            + config_seq.action_embedder.hidden_size
            + config_seq.reward_embedder.hidden_size
        )
        self.seq_model = SEQ_MODELS[config_seq.seq_model_config.name](
            input_size=rnn_input_size, **config_seq.seq_model_config.to_dict()
        )

        ## 2. build another obs+act branch
        # code only for image-based continuous action problems
        self.current_observ_embedder = deepcopy(image_encoder_fn())
        action_embedding_size = 32
        self.current_action_embedder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )
        shortcut_embedding_size = action_embedding_size + observ_embedding_size

        config_critic = config_rl.config_critic

        ## 4. build q networks
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.seq_model.hidden_size + shortcut_embedding_size + action_dim,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # policy network (takes in hidden state + shortcut)
        self.policy = self.algo.build_actor(
            input_size=self.seq_model.hidden_size + shortcut_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=config_rl.config_actor.hidden_dims,
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        assert config_rl.critic_lr == config_rl.actor_lr
        self.optimizer = Adam(self._get_parameters(), lr=config_rl.critic_lr)

        # pretrained history encoder
        obs_space = {"image": (obs_dim, 1, 1)}
        self.history_encoder = BeliefVAEModel(obs_space,
                                              env_name,
                                              config_critic.believer.x_dim,
                                              config_critic.believer.x_size,
                                              config_critic.believer.latent_dim)
        self.half_latent_dim = config_critic.believer.latent_dim // 2
        self.encoder_optimizer = torch.optim.Adam(
            self.history_encoder.parameters(),
            lr=config_critic.believer.lr
        )

        self.repr_model = RepresentationModel(env_name=env_name,
                                              state_space=obs_space,
                                              latent_dim=self.half_latent_dim)

        self.belief_agg = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.seq_model.hidden_size)
        )

        self.belief_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        if obs_embedder_dir is not None:
            self.history_encoder.load_state_dict(
                torch.load(obs_embedder_dir, map_location=ptu.device)
                )
            print(f"Believer: Loaded History Encoder: {obs_embedder_dir}")

        if state_embedder_dir is not None:
            self.repr_model.load_state_dict(
                torch.load(state_embedder_dir, map_location=ptu.device,
                           ),
                strict=False  # there are some unnecessary keys
            )
            self.repr_model.eval()
            print(f"Believer: Loaded State Encoder: {state_embedder_dir}")

    def _get_shortcut_obs_act_embedding(self, observs, current_actions):
        embed_observs = self.current_observ_embedder(observs)
        embed_actions = self.current_action_embedder(current_actions)
        return torch.cat([embed_observs, embed_actions], dim=-1)

    def _get_parameters(self):
        # exclude targets
        return [
            *self.observ_embedder.parameters(),
            *self.action_embedder.parameters(),
            *self.reward_embedder.parameters(),
            *self.seq_model.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
            *self.policy.parameters(),
        ]

    def get_hidden_states(
        self, observs,
        initial_internal_state=None
    ):
        if initial_internal_state is None:  # training
            initial_internal_state = self.history_encoder.get_zero_internal_state(
                batch_size=observs.shape[1]
            )  # initial_internal_state is zeros
            output, _ = self.history_encoder(observs.unsqueeze(-1).unsqueeze(-1),
                                             initial_internal_state)
            samples = self.history_encoder.sample(output)
            output = self.belief_agg(
                            torch.mean(self.belief_encoder(samples), dim=2)
                        )
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.history_encoder(
                observs.unsqueeze(-1).unsqueeze(-1), initial_internal_state
            )
            samples = self.history_encoder.sample(output)
            output = self.belief_agg(
                            torch.mean(self.belief_encoder(samples), dim=2)
                        )
            return output, current_internal_state


    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.history_encoder.get_zero_internal_state()
        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        state,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        obs = obs.unsqueeze(0)  # (1, B, dim)
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        hidden_state, current_internal_state = self.get_hidden_states(
            observs=obs,
            initial_internal_state=prev_internal_state,
        )
        curr_embed = self._get_shortcut_obs_act_embedding(
            obs, prev_action
        )  # (T+1, B, dim)
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        joint_embed = torch.cat(
            (hidden_state, curr_embed), dim=-1
        )  # (T, B, dim)

        if joint_embed.dim() == 3:
            joint_embed = joint_embed.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
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
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        # import time; t0 = time.time()
        hidden_states = self.get_hidden_states(
            observs=observs
        )
        curr_embed = self._get_shortcut_obs_act_embedding(
            observs, actions
        )  # (T+1, B, dim)
        # 3. joint embeds
        hidden_states = torch.cat(
            (hidden_states, curr_embed), dim=-1
        )  # (T+1, B, dim)
        # print("forward seq model", time.time() - t0)
        # NOTE: cost 30% time of single pass

        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim), new_next_log_probs: (T+1, B, 1 or A)
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(
                actor=self.policy,
                actor_target=self.policy_target,
                next_observ=hidden_states,
            )
            joint_q_embeds = torch.cat(
                (hidden_states, new_next_actions), dim=-1
            )  # (T+1, B, dim)

            next_q1 = self.qf1_target(joint_q_embeds)  # return (T, B, 1 or A)
            next_q2 = self.qf2_target(joint_q_embeds)
            min_next_q_target = torch.min(next_q1, next_q2)

            # min_next_q_target (T+1, B, 1 or A)
            min_next_q_target += self.algo.entropy_bonus(new_next_log_probs)
            if not self.algo.continuous_action:
                min_next_q_target = (new_next_actions * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )  # (T+1, B, 1)

            q_target = rewards + (1.0 - dones) * self.gamma * min_next_q_target
            q_target = q_target[1:]  # (T, B, 1)

        # Q(h(t), a(t)) (T, B, 1)
        # 3. joint embeds
        curr_joint_q_embeds = torch.cat(
            (hidden_states[:-1], actions[1:]), dim=-1
        )  # (T, B, dim)

        q1_pred = self.qf1(curr_joint_q_embeds)
        q2_pred = self.qf2(curr_joint_q_embeds)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        ### 3. Actor loss
        # Q(h(t), pi(h(t))) + H[pi(h(t))]
        # new_actions: (T+1, B, dim)
        new_actions, new_log_probs = self.algo.forward_actor(
            actor=self.policy, observ=hidden_states
        )

        if self.freeze_critic:
            ######## freeze critic parameters
            ######## and detach critic hidden states
            ######## such that the gradient only through new_actions
            new_joint_q_embeds = torch.cat(
                (hidden_states.detach(), new_actions), dim=-1
            )  # (T+1, B, dim)

            freezed_qf1 = deepcopy(self.qf1).to(ptu.device)
            freezed_qf2 = deepcopy(self.qf2).to(ptu.device)
            q1 = freezed_qf1(new_joint_q_embeds)
            q2 = freezed_qf2(new_joint_q_embeds)

        else:
            new_joint_q_embeds = torch.cat(
                (hidden_states, new_actions), dim=-1
            )  # (T+1, B, dim)

            q1 = self.qf1(new_joint_q_embeds)
            q2 = self.qf2(new_joint_q_embeds)

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1 or A)

        policy_loss = -min_q_new_actions
        policy_loss += -self.algo.entropy_bonus(new_log_probs)

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = (policy_loss * masks).sum() / num_valid

        ### 4. update
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss

        outputs = {
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
            "actor_loss": policy_loss.item(),
        }

        # import time; t0 = time.time()
        self.optimizer.zero_grad()
        total_loss.backward()
        # print("backward", time.time() - t0)
        # NOTE: cost 2/3 time of single pass

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm
            )
            outputs["raw_grad_norm"] = grad_norm.item()

        self.optimizer.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        if new_log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "seq_grad_norm": utl.get_grad_norm(self.seq_model),
            "critic_grad_norm": utl.get_grad_norm(self.qf1),
            "actor_grad_norm": utl.get_grad_norm(self.policy),
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

    def update_believer_encoder(self, batch):
        """
        Fine-tune the history encoder with interaction data
        """
        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)
        state, next_state = batch["state"], batch["state2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        states = torch.cat((state[[0]], next_state), dim=0)  # (T+1, B, dim)
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)

        # step 1: get state features
        state_features, _ = self.repr_model.encode_state(states)

        # step 2: calculate elbo loss
        prior_dist = Normal(0, 1)
        memory = self.history_encoder.get_zero_internal_state(
            batch_size=observs.shape[1]
        )
        hist_enc, memory = self.history_encoder(observs, memory)
        enc_mean, enc_std = self.history_encoder.encoder_dist(state_features,
                                                              hist_enc)
        zs = enc_mean + ptu.randn_like(enc_mean) * enc_std
        dec_mean, dec_std = self.history_encoder.decoder_dist(zs, hist_enc)
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        masks_ = torch.cat(
            (ptu.ones((1, observs.shape[1], 1)).float(),
             masks),
        )
        p1 = prior_dist.log_prob(zs)*masks_
        p2 = Normal(dec_mean, dec_std).log_prob(
            state_features)*masks_
        p3 = Normal(enc_mean, enc_std).log_prob(zs)*masks_
        elbo = p1.sum(dim=-1) + p2.sum(dim=-1) - p3.sum(dim=-1)
        loss = -elbo.sum() / (num_valid + 1)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()

        output = {}
        output["elbow_loss"] = loss.item()
        return output
