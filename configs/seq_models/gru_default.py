from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.is_markov = False
    config.is_attn = False
    config.use_dropout = False

    config.sampled_seq_len = -1

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False

    # fed into Module
    config.model = ConfigDict()

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "gru"
    config.model.seq_model_config.hidden_size = 128
    config.model.seq_model_config.n_layer = 1

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_size = 64

    config.model.current_observ_embedder = ConfigDict()
    config.model.current_observ_embedder.name = "mlp"
    config.model.current_observ_embedder.hidden_size = 32

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_size = 64

    config.model.reward_embedder = ConfigDict()
    config.model.reward_embedder.name = "mlp"
    config.model.reward_embedder.hidden_size = 0

    config.model.state_embedder = ConfigDict()
    config.model.state_embedder.name = "mlp"
    config.model.state_embedder.hidden_size = 32

    return config
