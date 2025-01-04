from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def dqn_name_fn(
    config: ConfigDict, max_episode_steps: int, max_training_steps: int
) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    # set eps = 1/T, so that the asymptotic prob to
    # sample fully exploited trajectory during exploration is
    # (1-1/T)^T = 1/e
    config.init_eps = 1.0
    config.end_eps = 1.0 / max_episode_steps
    config.schedule_steps = config.schedule_end * max_training_steps

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = dqn_name_fn

    config.algo = "dqn-ours"

    config.critic_lr = 3e-4

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)
    config.config_critic.config_recon = ConfigDict()
    config.config_critic.config_recon.enabled = True
    config.config_critic.config_recon.type = "recon_phi_s"
    config.config_critic.config_recon.hidden_act_fcn = "relu"
    config.config_critic.config_recon.output_act_fcn = "ident"
    config.config_critic.config_recon.hidden_dims = (128, 128)
    config.config_critic.config_recon.loss_fcn = "cosine"
    config.config_critic.config_recon.lr = 3e-4

    # intrinsic reward
    config.config_critic.config_recon.loss_w = 0.0
    config.config_critic.config_recon.reward_w = 10.0

    config.discount = 0.99
    config.tau = 0.005
    config.schedule_end = 0.1  # at least good for TMaze-like envs

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    return config
