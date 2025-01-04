import numpy as np
import random
import torch
import datetime
import dateutil.tz


def reproduce(seed):
    """
    This can only fix the randomness of numpy and torch
    To fix the environment's, please use
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    We have add these in our training script
    """
    assert seed >= 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reproduce_from_chkpt(chkpt_dict):
    """_summary_

    Args:
        chkpt_dict (_type_): _description_
    """
    random.setstate(chkpt_dict["random_rng_state"])
    np.random.set_state(chkpt_dict["np_rng_state"])
    torch.set_rng_state(chkpt_dict["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(chkpt_dict["torch_cuda_rng_state"])


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime(
        "%Y-%m-%d-%H:%M:%S"
    )  # may cause collision, please use PID to prevent
