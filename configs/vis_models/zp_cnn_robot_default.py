import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.channels = (32, 64, 64)
    config.kernel_sizes = (8, 4, 3)
    config.strides = (4, 2, 1)
    config.embedding_size = 112

    return config
