import os, time, socket

t0 = time.time()
pid = str(os.getpid())
if "SLURM_JOB_ID" in os.environ:
    jobid = str(os.environ["SLURM_JOB_ID"])
else:
    jobid = pid

import numpy as np
import joblib
import torch
from absl import app, flags
from ml_collections import config_flags
from utils import system, logger

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_repr",
    None,
    "File path to the representation model configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    "configs/seq_models/gru_default.py",
    "File path to the seq model configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_recon",
    None,
    "File path to the reconstruction model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl", "config_env"])

flags.DEFINE_boolean(
    "freeze_critic", False, "in shared encoder, freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("aux_batch_size", 64, "Mini batch size for aux tasks.")
flags.DEFINE_integer("train_episodes", 1000, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.25, "Gradient updates per step.")
flags.DEFINE_integer("sampled_seq_len", None, "Sampled sequence length.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")
flags.DEFINE_integer("start_expert_training", 0, "Number of expert episodes to start training.")
flags.DEFINE_integer("replay_buffer_size", 0, "Size of replay buffer")

# logging settings
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("run_name", None, "used in sbatch")
flags.DEFINE_boolean("save_csv", False, "save csv file")

# checkpoint in sbatch
flags.DEFINE_string("checkpoint_name", None, "checkpoint name")
flags.DEFINE_string("checkpoint_dir", None, "checkpoint directory")
flags.DEFINE_float("time_limit", 1000.0, "time limit (discovery)")

# cuda settings
flags.DEFINE_integer("cuda", 0, "gpu id")

# wandb settings
flags.DEFINE_string("group", "test", "wandb project name")

# replay policy
flags.DEFINE_string("policy_dir", None, "directory to the policy folder")

flags.DEFINE_string("env_name", None, "place holder for env name")

# state encoder
flags.DEFINE_string("state_embedder_dir", None, "state encoder directory")
flags.DEFINE_string("obs_embedder_dir", None, "observation encoder directory")
flags.DEFINE_float("critic_lr", None, "critic learning rate")
flags.DEFINE_float("actor_lr", None, "actor learning rate")


def main(argv):
    if FLAGS.seed < 0:
        seed = int(pid)  # to avoid conflict within a job which has same datetime
    else:
        seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq
    config_repr = FLAGS.config_repr
    config_recon = FLAGS.config_recon

    config_env, env_name = config_env.create_fn(config_env)
    replay_mode = FLAGS.policy_dir is not None
    tmp_env = make_env(env_name, seed, rendering=replay_mode)  # needed for replaying in robot domains
    config_seq, seq_name = config_seq.name_fn(config_seq, tmp_env.max_episode_steps)
    if FLAGS.checkpoint_name is None:
        checkpoint_name = f"{env_name}-{config_rl['algo']}-{seq_name[:-1]}-s{FLAGS.seed}"
        FLAGS.checkpoint_name = checkpoint_name

    if FLAGS.critic_lr is not None:
        config_rl["critic_lr"] = FLAGS.critic_lr

    if FLAGS.actor_lr is not None:
        config_rl["actor_lr"] = FLAGS.actor_lr

    if FLAGS.replay_buffer_size > 0:
        config_rl["replay_buffer_size"] = FLAGS.replay_buffer_size

    ckpt_dir = FLAGS.checkpoint_dir
    ckpt_name = FLAGS.checkpoint_name

    if ckpt_dir is not None:
        FLAGS.checkpoint_dir = os.path.join(ckpt_dir, ckpt_name)
    else:
        FLAGS.checkpoint_dir = ckpt_name

    ckpt_dict = None
    if os.path.exists(FLAGS.checkpoint_dir):  # checkpoint exists
        ckpt_dict = joblib.load(FLAGS.checkpoint_dir)
        print("Load checkpoint file done")

        system.reproduce_from_chkpt(ckpt_dict)
        if "Block" in env_name:
            env = make_env(env_name, seed)
            eval_env = make_env(env_name, seed + 42, replay_mode)
            env.reset()
            eval_env.reset()
        env = ckpt_dict["_train_env"]
        eval_env = ckpt_dict["_eval_env"]
    else:
        system.reproduce(seed)
        env = make_env(env_name, seed)
        eval_env = make_env(env_name, seed + 42, replay_mode)

    FLAGS.env_name = env_name

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    if socket.gethostname() in ['theseus', 'titan', 'taichi', 'cs-daedalus-bu-edu']:
        if socket.gethostname() in ['titan', 'taichi', 'cs-daedalus-bu-edu']:
            FLAGS.cuda = FLAGS.cuda % 2
        else:
            FLAGS.cuda = FLAGS.cuda % 4
    else:
        FLAGS.cuda = 0

    set_gpu_mode(torch.cuda.is_available(), FLAGS.cuda)

    if FLAGS.sampled_seq_len is not None:
        config_seq.sampled_seq_len = FLAGS.sampled_seq_len

    max_training_steps = int(FLAGS.train_episodes * env.max_episode_steps)
    config_rl, _ = config_rl.name_fn(config_rl, env.max_episode_steps, max_training_steps)
    uid = f"{system.now_str()}+{jobid}-{pid}"
    run_name = f"{config_env.env_type}/{seq_name}/{uid}"
    FLAGS.run_name = uid

    format_strs = ["wandb"]
    if FLAGS.debug:
        FLAGS.save_dir = "debug"
        format_strs = []
        format_strs.extend(["stdout", "log"])  # logger.log
    if FLAGS.save_csv:
        format_strs.extend(["csv"])

    log_path = os.path.join(FLAGS.save_dir, run_name)

    if not replay_mode:
        logger.configure(dir=log_path, format_strs=format_strs, config=FLAGS)

        # write flags to a txt
        key_flags = FLAGS.get_key_flags_for_module(argv[0])
        with open(os.path.join(log_path, "flags.txt"), "w") as text_file:
            text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")

    # start training
    learner = Learner(env, eval_env, FLAGS,
                      config_rl, config_seq, config_env,
                      ckpt_dict,
                      config_repr, config_recon,
                      )

    # print(learner.agent)
    print("Number of weights:", learner.num_of_weights())
    # GPT (2 layers, 2 heads): 1M
    # LSTM: 1 layer: 300K

    if replay_mode:
        learner.replay_policy(FLAGS.policy_dir)
    else:
        learner.train()


if __name__ == "__main__":
    app.run(main)
