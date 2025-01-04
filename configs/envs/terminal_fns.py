import gym


def finite_horizon_terminal(env: gym.Env, done: bool, info: dict) -> bool:
    return done


def infinite_horizon_terminal(env: gym.Env, done: bool, info: dict) -> bool:
    if "TimeLimit.truncated" in info:
        truncated = True
    else:
        truncated = False

    if done is False:
        terminal = False
    else:
        if truncated is True:
            terminal = False
        else:
            terminal = True
    return terminal
