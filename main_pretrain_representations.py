from absl import app, flags
from ml_collections import config_flags
import os

from policies.models.repr_model import ReprModel
from policies.models.train_repr_model import ReprTrainer
from envs.make_env import make_env

import torch
from torch_ac.utils import DictList
import torch.nn.functional as F
from torchkit import pytorch_utils as ptu

import wandb


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_repr",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)


flags.DEFINE_string("data_path", None, "Path to collected data.")
flags.DEFINE_integer("num_epochs", 10, "# of training epochs (default 10).")
flags.DEFINE_integer("seed", 0, "Random seed (default 0).")
flags.DEFINE_integer("log_freq", 50, "# of epochs to log file.")
flags.DEFINE_string("wandb_name", None, "Name of wandb run.")
flags.DEFINE_string("init_model", None, "Path to pretrained model to begin training from.")
flags.mark_flags_as_required(["config_env", "config_repr", "data_path"])


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_env = FLAGS.config_env
    config_env, env_name = config_env.create_fn(config_env)

    env = make_env(env_name, 0)
    state_dim = env.state_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    try:
        action_dim = env.action_space.shape[0]
    except:
        action_dim = env.action_space.n

    repr_model = ReprModel(state_dim=state_dim,
                           obs_dim=obs_dim,
                           action_dim=action_dim,
                           config=FLAGS.config_repr).to(device)

    if FLAGS.init_model is not None:
        repr_model.load_state_dict(torch.load(FLAGS.init_model,
                                              map_location=device))
    wandb.login(key='ed44c646a708f75a7fe4e39aee3844f8bfe44858')
    wandb.init(project="CV-MIM",
               settings=wandb.Settings(_disable_stats=True),
               group=f"{env_name}",
               name=f"s{FLAGS.seed}" if FLAGS.wandb_name is None else FLAGS.wandb_name,
               config=FLAGS.flag_values_dict(),
               entity='pomdpr')
    # Load data
    data = torch.load(FLAGS.data_path)
    eps_len = data["masks"].shape[1]
    indices = (data["masks"] == 1).nonzero(as_tuple=True)
    next_indices = (indices[0], indices[1]+1)
    next_indices_clamped = (indices[0], torch.clamp(indices[1]+1, max=eps_len-1))

    exps = DictList()

    exps.obs = data["obss"][indices].to(device)
    exps.state = data["states"][indices].to(device)
    exps.next_obs = data["obss"][next_indices_clamped].to(device)
    exps.next_state = data["states"][next_indices_clamped].to(device)

    exps.next_mask = F.pad(input=data["masks"], pad=(0, 1))[next_indices].to(device)
    exps.action = data["actions"][indices].to(device)
    exps.reward = data["rewards"][indices].to(device)

    trainer = ReprTrainer(env, exps, repr_model, FLAGS.config_repr)

    for epoch in range(FLAGS.num_epochs):
        kvs = trainer.update()
        for k, v in kvs.items():
            wandb.log({k: v}, step=epoch)

        if epoch % FLAGS.log_freq == 0 and epoch > 0:
            file_dir = f"pretrained_encoders/{env_name}/s{FLAGS.seed}_e{epoch}_original"
            os.makedirs(file_dir, exist_ok=True)
            repr_model.save(file_dir, to_cloud=True)

    file_dir = f"pretrained_encoders/{env_name}/s{FLAGS.seed}_end{epoch}_original"
    os.makedirs(file_dir, exist_ok=True)
    repr_model.save(file_dir, to_cloud=True)


if __name__ == "__main__":
    app.run(main)