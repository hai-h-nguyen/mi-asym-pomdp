from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu
from policies.models.state_reconstructor import State_Reconstructor, RECON_LOSS_FCNS
from policies.models.models_cv_mim import SimpleModel, SimpleModelCNN


class Actor_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        state_dim,
        config_seq,
        config_actor,
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

        ## 3. build another obs branch
        if self.image_encoder is None:
            self.current_observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:
            self.current_observ_embedder = deepcopy(image_encoder)

        ## 4. build policy
        # using psi(o) as well
        if self.config_repr is not None and obs_embedder_dir is not None:
            observ_embedding_size = self.config_repr.obs_embedding_dim

        self.policy = self.algo.build_actor(
            input_size=self.seq_model.hidden_size + observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=config_actor.hidden_dims,
        )

        ## 5. build state reconstructor
        self.recon_type = config_actor.config_recon.type
        self.recon = State_Reconstructor(
            config_actor.config_recon,
            config_repr.state_embedding_dim,
            state_dim,
            self.seq_model.hidden_size,
        )
        self.recon_optimizer = torch.optim.Adam(self.recon.parameters(),
                                                lr=config_actor.config_recon.lr)

        self.recon_loss_fcn = RECON_LOSS_FCNS[config_actor.config_recon.loss_fcn]
        self.recon_loss_w = config_actor.config_recon.loss_w
        self.recon_reward_w = config_actor.config_recon.reward_w

        if self.recon_type == "recon_phi_s":
            if self.image_size is not None:
                self.state_embedder = SimpleModelCNN(img_shape=self.image_size,
                                                     output_dim=self.config_repr.state_embedding_dim,
                                                     channels=self.config_repr.channels,
                                                     kernel_size=self.config_repr.kernel_sizes,
                                                     strides=self.config_repr.strides,
                                                     pool=self.config_repr.pool,
                                                     pool_size=self.config_repr.pool_size,
                                                     from_flattened=True)
                # self.state_embedder = deepcopy(image_encoder)
                if obs_embedder_dir is not None:
                    self.current_observ_embedder = SimpleModelCNN(img_shape=self.image_size,
                                                                  output_dim=self.config_repr.obs_embedding_dim,
                                                                  channels=self.config_repr.channels,
                                                                  kernel_size=self.config_repr.kernel_sizes,
                                                                  strides=self.config_repr.strides,
                                                                  pool=self.config_repr.pool,
                                                                  pool_size=self.config_repr.pool_size,
                                                                  from_flattened=True)
            else:
                assert self.config_repr is not None
                self.state_embedder = SimpleModel(state_dim,
                                                  self.config_repr.state_embedding_dim,
                                                  hidden_dim=self.config_repr.hidden_dim,
                                                  num_residual_linear_block=1,
                                                  num_layers_per_block=1,
                                                  dropout_rate=0.0,
                                                  use_batch_norm=False,
                                                  from_flattened=True)
                if obs_embedder_dir is not None:
                    self.current_observ_embedder = SimpleModel(obs_dim,
                                                               self.config_repr.obs_embedding_dim,
                                                               hidden_dim=self.config_repr.hidden_dim,
                                                               num_residual_linear_block=1,
                                                               num_layers_per_block=1,
                                                               dropout_rate=0.0,
                                                               use_batch_norm=False,
                                                               from_flattened=True)

            # Load weights
            if state_embedder_dir is not None:
                self.state_embedder.load_state_dict(
                    torch.load(state_embedder_dir, map_location=ptu.device)
                    )
                self.state_embedder.eval()
                print("Loaded State Encoder from: ", state_embedder_dir)

            if obs_embedder_dir is not None:
                self.current_observ_embedder.load_state_dict(
                    torch.load(obs_embedder_dir, map_location=ptu.device)
                    )
                self.current_observ_embedder.eval()
                print("Loaded Curr Obs Encoder from: ", obs_embedder_dir)

    def _get_shortcut_obs_embedding(self, observs):
        return self.current_observ_embedder(observs)

    def _calc_features(self, states):
        if self.image_size is not None:
            if states.ndim == 2:
                states = states.unsqueeze(1)
        out = self.state_embedder(states)
        return out

    @torch.no_grad()
    def get_state_labels(self, states):
        """Produce phi(s)

        Args:
            states (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.recon_type == "recon_s":
            return states

        if self.recon_type == "recon_phi_s":
            assert self.state_embedder is not None
            out = self._calc_features(states)
            return out

    @torch.no_grad()
    def calc_intrinsic_rewards(self, actions, rewards, observs, states, dones, masks):
        batch_size = actions.shape[1]

        hidden_states = self.get_hidden_states(
            prev_actions=actions, rewards=rewards,
            observs=observs,
        )
        recon_state = self.recon(hidden_states)
        target = self.get_state_labels(states)

        recon_masks = torch.cat(
            (ptu.ones((1, batch_size, 1)).float(),
             masks),
        )
        actor_recon_s = recon_state * recon_masks
        target = target * recon_masks
        intrinsic_r = recon_masks.squeeze(-1) - torch.abs(self.recon_loss_fcn(actor_recon_s, target))
        delta_r = intrinsic_r[:-1, :] - intrinsic_r[1:, :]
        delta_r *= self.recon_reward_w
        delta_r = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(),
             delta_r.unsqueeze(-1)),
            dim=0
        )  # (T+1, B, dim)
        return delta_r

    def get_hidden_states(
        self, prev_actions, rewards, observs,
        initial_internal_state=None
    ):
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self.observ_embedder(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1]
            )  # initial_internal_state is zeros
            output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout
            if isinstance(initial_internal_state, int):
                initial_internal_state = None
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
            return output, current_internal_state

    def update_recon(self, prev_actions, rewards, observs, states, masks):
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards,
            observs=observs,
        )
        recon_state = self.recon(hidden_states.detach())
        recon_target = self.get_state_labels(states)
        batch_size = observs.shape[1]
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        recon_masks = torch.cat(
            (ptu.ones((1, batch_size, 1)).float(),
             masks),
        )
        actor_recon_s = recon_state * recon_masks
        target = recon_target * recon_masks
        recon_loss = (recon_masks.squeeze(-1) -
                      torch.abs(self.recon_loss_fcn(actor_recon_s, target))).sum()
        recon_loss /= (num_valid + 1)

        self.recon_optimizer.zero_grad()
        recon_loss.backward()
        self.recon_optimizer.step()

        output = {}
        output["actor_recon_loss"] = recon_loss.item()
        return output

    def forward(self, prev_actions, rewards, observs):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == rewards.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards,
            observs=observs,
        )

        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(observs)  # (T+1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)  # (T+1, B, dim)

        aux_info = {}
        # recon_state = self.recon(hidden_states)
        # aux_info["recon_state"] = recon_state
        # aux_info["recon_loss_fcn"] = self.recon_loss_fcn
        # aux_info["recon_loss_w"] = self.recon_loss_w
        # aux_info["recon_target"] = self.get_state_labels(states)

        return self.algo.forward_actor(actor=self.policy, observ=joint_embeds), aux_info

    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.seq_model.get_zero_internal_state()

        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observs=obs,
            initial_internal_state=prev_internal_state,
        )

        # 2. another branch for current obs
        curr_embed = self._get_shortcut_obs_embedding(obs)  # (1, B, dim)

        # 3. joint embed
        joint_embed = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)

        if joint_embed.dim() == 3:
            joint_embed = joint_embed.squeeze(0)  # (B, dim)

        current_action = self.algo.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return current_action, current_internal_state
