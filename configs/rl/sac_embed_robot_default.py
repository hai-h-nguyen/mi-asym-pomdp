from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def sac_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name + f"alpha-{config.init_temperature}/"


def get_config():
    config = ConfigDict()
    config.name_fn = sac_name_fn

    config.algo = "sac-embed"

    config.actor_lr = 1e-3
    config.critic_lr = 1e-3
    config.temp_lr = 1e-3

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)
    config.config_actor.expert_w = 1.0

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005

    config.automatic_entropy_tuning = True
    config.init_temperature = 0.01
    config.entropy_alpha = 0.1

    config.replay_buffer_size = 1e5
    config.replay_buffer_num_episodes = 1e3

    return config
