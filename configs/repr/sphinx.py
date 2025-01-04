from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    # training specific
    config.lr_encoder = 1e-3
    config.lr_estimator = 1e-3
    config.dynamics_loss_s_weight = 0.5
    config.dynamics_loss_o_weight = 0.03
    config.reward_loss_weight = 10.0
    config.representation_loss_o_weight = 0.5
    config.disentangle_loss_weight = 0.3

    config.adam_eps = 1e-8
    config.batch_size = 500

    # fed into Module
    config.state_embedding_dim = 32
    config.obs_embedding_dim = 32
    config.act_embedding_dim = 16
    config.hidden_dim = 256
    config.channels = [64, 128, 256]
    config.kernel_sizes = [2, 2, 2]
    config.strides = [1, 1, 1]
    config.pool = [False, True, False]
    config.pool_size = [0, 2, 0]
    config.use_cnn = True

    return config
