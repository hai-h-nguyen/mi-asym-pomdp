from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl


class Actor_Embed(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_actor,
        algo,
        image_encoder=None,
        image_size=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.seq_model_name = config_seq.seq_model_config.name
        self.image_size = image_size

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        self.image_encoder = image_encoder
        if image_encoder is None:
            observ_embedding_size = config_seq.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = deepcopy(image_encoder)
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.policy = self.algo.build_actor(
            input_size=observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=config_actor.hidden_dims,
        )

    def forward(self, observs):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert observs.dim() == 2

        curr_embed = self.observ_embedder(observs)  # (T+1, B, dim)

        return self.algo.forward_actor(actor=self.policy, observ=curr_embed)

    @torch.no_grad()
    def act(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        curr_embed = self.observ_embedder(obs)  # (1, B, dim)

        current_action = self.algo.select_action(
            actor=self.policy,
            observ=curr_embed,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return current_action
