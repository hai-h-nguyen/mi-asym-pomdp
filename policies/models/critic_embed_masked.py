from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl


class Critic_Embed(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_critic,
        algo,
        image_encoder=None,
        image_size=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
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
            self.state_embedder = deepcopy(image_encoder)
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, config_seq.action_embedder.hidden_size, F.relu
        )

        ## 4. build q networks
        qf = self.algo.build_critic(
            input_size=config_seq.action_embedder.hidden_size + observ_embedding_size,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )
        if isinstance(qf, tuple):
            self.qf1, self.qf2 = qf
        else:
            self.qf = qf

    def _get_shortcut_obs_act_embedding(self, observs, current_actions):
        # reshape
        # observs = observs.view(-1, 2, 84, 84)
        # depth = observs.clone()
        # depth[:, [1], :, :] = 0
        # depth = depth.view(-1, 2*84*84)

        # rgb_blue = observs.clone()
        # rgb_blue[:, [0], :, :] = 0
        # rgb_blue = rgb_blue.view(-1, 2*84*84)

        embed_observs = self.observ_embedder(observs)
        # embed_states = self.state_embedder(rgb_blue)
        embed_actions = self.action_embedder(current_actions)
        return torch.cat([embed_observs, embed_actions], dim=-1)

    def forward(self, observs, current_actions):
        """
        For prev_actions a, rewards r, observs o: (T, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """

        # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
        curr_embed = self._get_shortcut_obs_act_embedding(
            observs, current_actions
        )  # (T+1, B, dim)

        q1 = self.qf1(curr_embed)
        q2 = self.qf2(curr_embed)

        return q1, q2  # (T, B, 1 or A)
