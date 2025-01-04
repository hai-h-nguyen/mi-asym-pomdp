import torch
import torch.nn as nn
from policies.models.models_cv_mim import SimpleModel, SimpleModelCNN
import torchkit.pytorch_utils as ptu


class Actor(nn.Module):
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
        self.zero_bottom = config_actor.zero_bottom

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if self.image_encoder is not None:
            self.state_embedder = SimpleModelCNN(img_shape=self.image_size,
                                                 output_dim=self.config_repr.state_embedding_dim,
                                                 channels=self.config_repr.channels,
                                                 kernel_size=self.config_repr.kernel_sizes,
                                                 strides=self.config_repr.strides,
                                                 pool=self.config_repr.pool,
                                                 pool_size=self.config_repr.pool_size,
                                                 from_flattened=True)
            if obs_embedder_dir is not None:
                self.observ_embedder = SimpleModelCNN(img_shape=self.image_size,
                                                      output_dim=self.config_repr.obs_embedding_dim,
                                                      channels=self.config_repr.channels,
                                                      kernel_size=self.config_repr.kernel_sizes,
                                                      strides=self.config_repr.strides,
                                                      pool=self.config_repr.pool,
                                                      pool_size=self.config_repr.pool_size,
                                                      from_flattened=True)
        else:
            self.state_embedder = SimpleModel(state_dim,
                                              self.config_repr.state_embedding_dim,
                                              hidden_dim=self.config_repr.hidden_dim,
                                              num_residual_linear_block=1,
                                              num_layers_per_block=1,
                                              dropout_rate=0.0,
                                              use_batch_norm=False,
                                              from_flattened=True)
            if obs_embedder_dir is not None:
                self.observ_embedder = SimpleModel(obs_dim,
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
            self.observ_embedder.load_state_dict(
                torch.load(obs_embedder_dir, map_location=ptu.device)
                )
            self.observ_embedder.eval()
            print("Loaded Curr Obs Encoder from: ", obs_embedder_dir)

        input_size = 0
        input_size += self.config_repr.state_embedding_dim
        input_size += self.config_repr.obs_embedding_dim
        self.policy = self.algo.build_actor(
            input_size=input_size,
            action_dim=self.action_dim,
            hidden_sizes=config_actor.hidden_dims,
        )

    def _get_embedding(self, observs, states):
        states = states.unsqueeze(1)
        observs = observs.unsqueeze(1)
        with torch.no_grad():
            embed_observs = self.observ_embedder(observs)
            embed_states = self.state_embedder(states)
        if self.zero_bottom:
            embed_observs = ptu.zeros_like(embed_observs)
        out = torch.cat([embed_observs, embed_states], dim=-1)
        out = out.squeeze(1)
        return out

    def forward(self, observs, states):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert observs.dim() == states.dim() == 2

        embed = self._get_embedding(observs, states)

        return self.algo.forward_actor(actor=self.policy, observ=embed)

    @torch.no_grad()
    def act(
        self,
        obs,
        state,
        deterministic=False,
        return_log_prob=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        embed = self._get_embedding(obs, state)

        current_action = self.algo.select_action(
            actor=self.policy,
            observ=embed,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return current_action
