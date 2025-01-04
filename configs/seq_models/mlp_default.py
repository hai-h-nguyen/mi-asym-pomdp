from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.is_markov = True
    config.is_attn = False

    config.sampled_seq_len = 1

    # fed into Module
    config.model = ConfigDict()

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "mlp"

    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_size = 32

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_size = 32

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False

    return config
