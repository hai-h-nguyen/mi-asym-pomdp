from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from utils import helpers as utl
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu
from policies.models.believer_encoders import BeliefVAEModelGPT, RepresentationModel


class Critic_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        env_name,
        config_seq,
        config_critic,
        config_repr,
        algo,
        state_embedder_dir=None,
        obs_embedder_dir=None,
        image_encoder=None,
        image_size=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config_repr = config_repr
        self.algo = algo
        self.image_encoder = image_encoder
        self.seq_model_name = config_seq.seq_model_config.name
        self.image_size = image_size

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder is None:
            observ_embedding_size = config_seq.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = deepcopy(image_encoder)
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

        ## 3. build another obs+act branch
        shortcut_embedding_size = rnn_input_size
        if self.algo.continuous_action and self.image_encoder is None:
            # for vector-based continuous action problems
            self.current_observ_embedder = utl.FeatureExtractor(
                obs_dim, shortcut_embedding_size, F.relu
            )
            self.current_action_embedder = utl.FeatureExtractor(
                action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += shortcut_embedding_size

        elif self.algo.continuous_action and self.image_encoder is not None:
            # for image-based continuous action problems
            self.current_observ_embedder = deepcopy(image_encoder)
            self.current_action_embedder = utl.FeatureExtractor(
                action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += observ_embedding_size
        elif not self.algo.continuous_action and self.image_encoder is None:
            # for vector-based discrete action problems
            self.current_observ_embedder = utl.FeatureExtractor(
                obs_dim, shortcut_embedding_size, F.relu
            )
        elif not self.algo.continuous_action and self.image_encoder is not None:
            # for image-based discrete action problems
            self.current_observ_embedder = deepcopy(image_encoder)
            shortcut_embedding_size = observ_embedding_size
        else:
            raise NotImplementedError

        ## 4. build q networks
        qf = self.algo.build_critic(
            input_size=self.seq_model.hidden_size + shortcut_embedding_size,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )
        if isinstance(qf, tuple):
            self.qf1, self.qf2 = qf
        else:
            self.qf = qf

        # pretrained history encoder
        obs_space = {"image": (obs_dim, 1, 1)}
        self.history_encoder = BeliefVAEModelGPT(obs_space,
                                                 env_name,
                                                 config_critic.believer.x_dim,
                                                 config_critic.believer.x_size,
                                                 config_seq,
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
        if self.algo.continuous_action:
            embed_observs = self.current_observ_embedder(observs)
            embed_actions = self.current_action_embedder(current_actions)
            return torch.cat([embed_observs, embed_actions], dim=-1)
        else:
            return self.current_observ_embedder(observs)

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
                observs, initial_internal_state
            )
            samples = self.history_encoder.sample(output)
            output = self.belief_agg(
                            torch.mean(self.belief_encoder(samples), dim=2)
                        )
            return output, current_internal_state

    def update_believer_encoder(self, observs, states, masks):
        """
        Fine-tune the history encoder with interaction data
        """
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

    def forward(self, prev_actions, rewards, observs, current_actions):
        """
        For prev_actions a, rewards r, observs o: (T, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert prev_actions.dim() == rewards.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        hidden_states = self.get_hidden_states(
            observs=observs,
        )

        # 2. another branch for state & **current** action
        if self.algo.continuous_action:
            if current_actions.shape[0] == observs.shape[0]:
                # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
                curr_embed = self._get_shortcut_obs_act_embedding(
                    observs, current_actions
                )  # (T+1, B, dim)
                # 3. joint embeds
                joint_embeds = torch.cat(
                    (hidden_states, curr_embed), dim=-1
                )  # (T+1, B, dim)
            else:
                # current_actions does NOT include last obs's action
                curr_embed = self._get_shortcut_obs_act_embedding(
                    observs[:-1], current_actions
                )  # (T, B, dim)
                # 3. joint embeds
                joint_embeds = torch.cat(
                    (hidden_states[:-1], curr_embed), dim=-1
                )  # (T, B, dim)
        else:
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs, current_actions
            )  # (T+1, B, dim)
            joint_embeds = torch.cat(
                (hidden_states, curr_embed), dim=-1
            )  # (T, B, dim)

        # q value
        if hasattr(self, "qf"):
            q = self.qf(joint_embeds)
            return q
        else:
            q1 = self.qf1(joint_embeds)
            q2 = self.qf2(joint_embeds)
            return q1, q2  # (T, B, 1 or A)

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
        obs,
        deterministic=False,
    ):
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

        current_action = self.algo.select_action(
            qf=self.qf,  # assume single q head
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
