from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    # training specific
    config.lr_encoder = 1e-3
    config.lr_estimator = 1e-3
    config.dynamics_loss_s_weight = 1.0
    config.dynamics_loss_o_weight = 1.0
    config.reward_loss_weight = 100.0
    config.representation_loss_s_weight = 0.01
    config.representation_loss_o_weight = 1.0
    config.disentangle_loss_weight = 1e-3

    config.adam_eps = 1e-8
    config.batch_size = 500

    # fed into Module
    config.state_embedding_dim = 32
    config.obs_embedding_dim = 32
    config.act_embedding_dim = 16
    config.hidden_dim = 1024
    config.channels = [32, 64, 64]
    config.kernel_sizes = [8, 4, 3]
    config.strides = [4, 4, 1]
    config.pool = [False, False, False]
    config.pool_size = [0, 0, 0]
    config.use_cnn = True

    return config
