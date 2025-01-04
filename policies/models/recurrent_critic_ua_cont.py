from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu


class Critic_RNN(nn.Module):
    """Used for UA-SAC (continuous action)

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        obs_dim,
        action_dim,
        state_dim,
        config_seq,
        config_critic,
        algo,
        image_encoder=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.image_encoder = image_encoder
        self.seq_model_name = config_seq.seq_model_config.name

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder is None:
            observ_embedding_size = config_seq.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
            self.state_embedder = utl.FeatureExtractor(
                state_dim,
                config_seq.state_embedder.hidden_size, F.relu)
        else:  # for pixel observation, use external encoder
            self.observ_embedder = deepcopy(image_encoder)
            self.state_embedder = deepcopy(image_encoder)
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
        if self.image_encoder is None:
            # for vector-based continuous action problems
            self.current_observ_embedder = utl.FeatureExtractor(
                obs_dim, shortcut_embedding_size, F.relu
            )
            self.current_action_embedder = utl.FeatureExtractor(
                action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += shortcut_embedding_size

        else:
            # for image-based continuous action problems
            self.current_observ_embedder = deepcopy(image_encoder)
            self.current_action_embedder = utl.FeatureExtractor(
                action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += self.image_encoder.embedding_size

        ## 4. build q networks
        # qhz model
        if image_encoder is None:
            shortcut_embedding_size += config_seq.state_embedder.hidden_size
        else:
            shortcut_embedding_size += observ_embedding_size

        qzf = self.algo.build_critic(
            input_size=self.seq_model.hidden_size + shortcut_embedding_size,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )

        self.qzf1, self.qzf2 = qzf

    def _get_shortcut_obs_act_embedding(self, observs, current_actions):
        embed_observs = self.current_observ_embedder(observs)
        embed_actions = self.current_action_embedder(current_actions)
        return torch.cat([embed_observs, embed_actions], dim=-1)

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

    def forward(self, prev_actions, rewards, observs, states, current_actions):
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
            prev_actions=prev_actions, rewards=rewards, observs=observs,
        )

        # 2. another branch for state & **current** action
        if current_actions.shape[0] == observs.shape[0]:
            # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs, current_actions
            )  # (T+1, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states, curr_embed), dim=-1
            )  # (T+1, B, dim)
            state_embeds = self.state_embedder(states)
        else:
            # current_actions does NOT include last obs's action
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs[:-1], current_actions
            )  # (T, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states[:-1], curr_embed), dim=-1
            )  # (T, B, dim)
            state_embeds = self.state_embedder(states[:-1])

        joint_state_embeds = torch.cat(
            (joint_embeds, state_embeds), dim=-1
        )

        # q values
        qzf1 = self.qzf1(joint_state_embeds)
        qzf2 = self.qzf2(joint_state_embeds)

        return qzf1, qzf2

    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.seq_model.get_zero_internal_state()

        return prev_action, reward, internal_state
