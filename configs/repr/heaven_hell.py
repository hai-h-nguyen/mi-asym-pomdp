from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    # training specific
    config.lr_encoder = 3e-3
    config.lr_estimator = 3e-4
    config.reward_loss_weight = 10.0
    config.dynamics_loss_s_weight = 1.0
    config.dynamics_loss_o_weight = 0.5
    config.disentangle_loss_weight = 1.0
    config.representation_loss_s_weight = 0.0
    config.representation_loss_o_weight = 1.0

    config.adam_eps = 1e-8
    config.batch_size = 500

    # fed into Module
    config.state_embedding_dim = 32
    config.obs_embedding_dim = 32
    config.act_embedding_dim = 16
    config.hidden_dim = 128
    config.use_batch_norm = False
    config.dropout_rate = 0.0
    config.use_cnn = False

    return config