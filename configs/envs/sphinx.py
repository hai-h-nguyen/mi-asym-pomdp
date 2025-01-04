from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = "Sphinx-v0"
    register(
        env_name,
        entry_point="envs.sphinx:SphinxEnv",
        max_episode_steps=50,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "sphinx_v0"
    config.terminal_fn = finite_horizon_terminal  # this should be infinite, but finite works better

    config.eval_interval = 10
    config.save_interval = -1  # save model frequency
    config.eval_episodes = 10

    return config
