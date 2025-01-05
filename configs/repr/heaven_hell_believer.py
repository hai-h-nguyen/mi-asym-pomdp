from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    # training specific
    config.lr = 0.001
    config.beta = 1.0
    config.dynamics_loss_s_coef = 1.0
    config.dynamics_loss_o_coef = 0.5
    config.reward_loss_coef = 10.0

    config.adam_eps = 1e-8
    config.batch_size = 128
    config.batch_size = 128

    # fed into Module
    config.state_embedding_dim = 32
    config.obs_embedding_dim = 32
    config.act_embedding_dim = 16
    config.hidden_dim = 128
    config.use_batch_norm = False
    config.dropout_rate = 0.0
    config.use_cnn = False

    return config

