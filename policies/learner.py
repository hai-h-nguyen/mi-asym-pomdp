import os
import time
import random
import joblib
import wandb

import math
import numpy as np
import torch
from torch.nn import functional as F

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder
from .models.models_cv_mim import *

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer
from buffers.simple_replay_buffer_aug import SimpleReplayBufferAug

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer
from buffers.seq_replay_buffer_efficient_aug import RAMEfficient_SeqReplayBufferAug

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import logger


class Learner:
    def __init__(
        self,
        env,
        eval_env,
        FLAGS,
        config_rl,
        config_seq,
        config_env,
        ckpt_dict,
        config_repr=None,
        config_recon=None,
    ):
        self.train_env = env
        self.eval_env = eval_env
        self.FLAGS = FLAGS

        self.config_rl = config_rl
        self.config_seq = config_seq
        self.config_env = config_env
        self.config_repr = config_repr
        self.config_recon = config_recon

        self.save_interval = config_env["save_interval"]
        self.time_limit = FLAGS.time_limit
        self.state_embedder_dir = FLAGS.state_embedder_dir
        self.obs_embedder_dir = FLAGS.obs_embedder_dir

        self.checkpoint_dir = FLAGS.checkpoint_dir
        self.checkpoint_dir_end = self.checkpoint_dir + "_end"
        self.handle_chkpt(ckpt_dict)

        if self.FLAGS.policy_dir is None:
            self.init_wandb(FLAGS)

        self.init_env()

        self.init_agent()

        self.init_train()

    def handle_chkpt(self, ckpt_dict):
        if os.path.exists(self.checkpoint_dir):
            self.chkpt_dict = ckpt_dict
        else:
            print("Checkpoint file not found")
            # this end file exists, then this run has ended
            if os.path.exists(self.checkpoint_dir_end):
                print("End file found, exit")
                exit(0)
            self.chkpt_dict = None

    def num_of_weights(self):
        return sum(p.numel() for p in self.agent.parameters() if p.requires_grad)

    def init_env(
        self,
    ):
        # get action / observation dimensions
        assert len(self.train_env.observation_space.shape) == 1  # flatten
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.state_dim = self.train_env.state_space.shape[0]

        if self.chkpt_dict is None:
            print(
                "obs_dim",
                self.obs_dim,
                "act_dim",
                self.act_dim,
                "state_dim",
                self.state_dim,
            )

    def init_agent(
        self,
    ):
        # initialize agent
        if self.config_seq.is_markov:
            if self.config_rl.algo == "dqn":
                agent_class = AGENT_CLASSES["Policy_DQN_MLP"]
            elif self.config_rl.algo == "dqn-on-off":
                agent_class = AGENT_CLASSES["Policy_DQN_MLP_On_Off"]
            else:
                if self.config_rl.algo == "sac-embed":
                    agent_class = AGENT_CLASSES["Policy_MLP_Embed"]
                elif self.config_rl.algo == "sac-on-off":
                    agent_class = AGENT_CLASSES["Policy_MLP_On_Off"]
                else:
                    agent_class = AGENT_CLASSES["Policy_MLP"]
        else:
            if self.config_rl.algo == "dqn":
                agent_class = AGENT_CLASSES["Policy_DQN_RNN"]
            elif self.config_rl.algo == "dqn-zp":
                agent_class = AGENT_CLASSES["Policy_DQN_RNN_ZP"]
            elif self.config_rl.algo == "dqn-ours":
                agent_class = AGENT_CLASSES["Policy_DQN_RNN_Ours"]
            elif self.config_rl.algo == "dqn-believer-rnn":
                agent_class = AGENT_CLASSES["Policy_DQN_RNN_Believer"]
            elif self.config_rl.algo == "dqn-believer":
                agent_class = AGENT_CLASSES["Policy_DQN_GPT_Believer"]
            elif self.config_rl.algo == "dqn-ua":  # unbiased
                agent_class = AGENT_CLASSES["Policy_DQN_RNN_UA"]
            elif self.config_rl.algo == "dqn-ba":  # biased
                agent_class = AGENT_CLASSES["Policy_DQN_RNN_BA"]

            elif self.config_rl.algo == "sac-ua":
                agent_class = AGENT_CLASSES["Policy_Separate_RNN_UA"]
            elif self.config_rl.algo == "sac-zp":
                agent_class = AGENT_CLASSES["Policy_Separate_RNN_ZP"]
            elif self.config_rl.algo == "sac-believer-rnn":
                agent_class = AGENT_CLASSES["Policy_Shared_RNN_Believer"]
            elif self.config_rl.algo == "sac-believer":
                agent_class = AGENT_CLASSES["Policy_Shared_GPT_Believer"]
            elif self.config_rl.algo == "sac-ours":
                agent_class = AGENT_CLASSES["Policy_Separate_RNN_Ours"]
            elif self.config_rl.algo == "sac-ba":
                agent_class = AGENT_CLASSES["Policy_Separate_RNN_BA"]
            else:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]

        self.agent_arch = agent_class.ARCH
        if self.chkpt_dict is None:
            logger.log(agent_class, self.agent_arch)

        if self.config_seq.model.observ_embedder.name == "cnn":
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape,
                normalize_pixel=(self.train_env.observation_space.dtype == np.uint8),
                **self.config_seq.model.observ_embedder.to_dict(),
            )
            image_size = self.train_env.image_space.shape
        else:
            image_encoder_fn = lambda: None
            image_size = None
        self.agent = agent_class(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            config_seq=self.config_seq,
            config_rl=self.config_rl,
            config_repr=self.config_repr,
            config_recon=self.config_recon,
            image_encoder_fn=image_encoder_fn,
            env_name=self.FLAGS.env_name,
            state_embedder_dir=self.state_embedder_dir,
            obs_embedder_dir=self.obs_embedder_dir,
            image_size=image_size,
            freeze_critic=self.FLAGS.freeze_critic,
        ).to(ptu.device)
        if self.chkpt_dict is not None:
            self.agent.restore_state_dict(self.chkpt_dict["agent_dict"])
            print("Load agent checkpoint done")

        if self.chkpt_dict is None:
            logger.log(self.agent)

    def init_train(
        self,
    ):

        if self.agent_arch == AGENT_ARCHS.Markov:
            if self.config_env.env_type == "robot":
                buffer_class = SimpleReplayBufferAug
                num_augs = 4
            else:
                buffer_class = SimpleReplayBuffer
                num_augs = 0

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(self.config_rl.replay_buffer_size),
                observation_dim=self.obs_dim,
                state_dim=self.state_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.train_env.max_episode_steps,
                num_augs=num_augs,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if self.config_env.env_type == "robot":
                buffer_class = RAMEfficient_SeqReplayBufferAug
                num_augs = 4
            else:
                buffer_class = RAMEfficient_SeqReplayBuffer
                num_augs = 0

            self.policy_storage = buffer_class(
                max_replay_buffer_size=max(
                    int(self.config_rl.replay_buffer_size),
                    int(
                        self.config_rl.replay_buffer_num_episodes
                        * self.train_env.max_episode_steps
                    ),
                ),
                observation_dim=self.obs_dim,
                state_dim=self.state_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=self.config_seq.sampled_seq_len,
                observation_type=self.train_env.observation_space.dtype,
                num_augs=num_augs,
            )

            if self.chkpt_dict is None:
                logger.log(buffer_class)

        # load buffer from checkpoint
        if self.chkpt_dict is not None:
            self.policy_storage.load_from_state_dict(self.chkpt_dict["buffer_dict"])
            print("Load buffer checkpoint done")

        total_rollouts = self.FLAGS.start_training + self.FLAGS.train_episodes
        self.n_env_steps_total = self.train_env.max_episode_steps * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_wandb(
        self,
        flags,
    ):
        wandb.login(key="ed44c646a708f75a7fe4e39aee3844f8bfe44858")
        seed = flags.seed
        env_name = flags.env_name

        wandb_args = {}
        if self.chkpt_dict is not None:
            wandb_args = {"resume": "allow", "id": self.chkpt_dict["wandb_id"]}
        else:
            wandb_args = {"resume": None}

        wandb.init(
            project=f"{env_name}",
            settings=wandb.Settings(_disable_stats=True),
            group=str(flags.group),
            name=f"s{seed}",
            config=flags.flag_values_dict(),
            entity="pomdpr",
            **wandb_args,
        )

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._last_time = 0

        self._best_per = 0.0

        if self.chkpt_dict is not None:
            self._n_env_steps_total = self.chkpt_dict["_n_env_steps_total"]
            self._n_rollouts_total = self.chkpt_dict["_n_rollouts_total"]
            self._n_rl_update_steps_total = self.chkpt_dict["_n_rl_update_steps_total"]
            self._n_env_steps_total_last = self.chkpt_dict["_n_env_steps_total_last"]
            self._last_time = self.chkpt_dict["_last_time"]
            self._best_per = self.chkpt_dict["_best_per"]
            print("Load training statistic done")
            # final load so save some memory
            self.chkpt_dict = {}

        self._start_time = time.time()
        self._start_time_last = time.time()

    def save_checkpoint(self):
        # Save checkpoint
        print(f"Saving checkpoint {self.checkpoint_dir}...")

        # save replay buffer data
        buffer_dict = self.policy_storage.get_state_dict()

        self._last_time += (time.time() - self._start_time) / 3600.0

        joblib.dump(
            {
                "buffer_dict": buffer_dict,
                "agent_dict": self.agent.get_state_dict(),
                "random_rng_state": random.getstate(),
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(),
                "_n_env_steps_total": self._n_env_steps_total,
                "_n_rollouts_total": self._n_rollouts_total,
                "_n_rl_update_steps_total": self._n_rl_update_steps_total,
                "_n_env_steps_total_last": self._n_env_steps_total_last,
                "wandb_id": wandb.run.id,
                "_last_time": self._last_time,
                "_best_per": self._best_per,
                "_train_env": self.train_env,
                "_eval_env": self.eval_env,
            },
            self.checkpoint_dir,
        )

    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.FLAGS.start_expert_training > 0 and self.chkpt_dict is None:
            print("Collecting expert pool of data..")
            self.collect_expert_rollouts(
                num_rollouts=self.FLAGS.start_expert_training,
                mdp_mode=self.config_rl.algo in ["sac-on-off", "sac-embed"],
            )
            print(
                "Done! expert env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            train_stats = self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )
            self.log_train_stats(train_stats)

        if self.FLAGS.start_training > 0 and self.chkpt_dict is None:
            print("Collecting initial pool of data..")
            while (
                self._n_env_steps_total
                < self.FLAGS.start_training * self.train_env.max_episode_steps
            ):
                self.collect_rollouts(
                    num_rollouts=1,
                    random_actions=True,
                )
            print(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            train_stats = self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )
            self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            if (time.time() - self._start_time) / 3600.0 > self.time_limit:
                self.save_checkpoint()
                print("Checkpointing done and exit")
                exit(0)

            env_steps = self.collect_rollouts(num_rollouts=1)
            logger.log("env steps", self._n_env_steps_total)

            train_stats = self.update(
                int(math.ceil(self.FLAGS.updates_per_step * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.train_env.max_episode_steps
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.config_env.eval_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()

                # save best model
                if (
                    self.save_interval > 0
                    and self._n_env_steps_total >= 0.75 * self.n_env_steps_total
                ):
                    if perf > self._best_per:
                        print(f"Replace {self._best_per} w/ {perf} model")
                        self._best_per = perf
                        self.save_model(current_num_iters, perf, to_cloud=True)

                # save model according to a frequency
                if (
                    self.save_interval > 0
                    and self._n_env_steps_total >= 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf, to_cloud=True)
        self.save_model(current_num_iters, perf, to_cloud=True)

        if os.path.exists(self.checkpoint_dir):
            # remove checkpoint file to save space
            os.system(f"rm {self.checkpoint_dir}")
            print("Remove checkpoint file")

            # create a file to signify that this run has ended
            joblib.dump(
                {"_n_env_steps_total": self._n_env_steps_total}, self.checkpoint_dir_end
            )
            print("Created end file")

    def replay_policy(self, policy_dir):
        """
        replay a policy
        """
        self.load_model(policy_dir)
        self.evaluate()
        _, success_rate_eval, total_steps_eval = self.evaluate()
        print("num episodes", len(total_steps_eval))
        print("success", np.mean(success_rate_eval))
        print("length", np.mean(total_steps_eval))
        print("max length", self.train_env.max_episode_steps)

    @torch.no_grad()
    def collect_expert_rollouts(self, num_rollouts, mdp_mode=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        expert_ep_cnt = 0
        while expert_ep_cnt < num_rollouts:
            steps = 0
            obs = ptu.from_numpy(self.train_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            state = ptu.from_numpy(self.train_env.get_state())  # state at reset
            state = state.reshape(1, state.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                (
                    obs_list,
                    act_list,
                    rew_list,
                    next_obs_list,
                    term_list,
                    state_list,
                    next_state_list,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            while not done_rollout:
                action = ptu.FloatTensor(
                    [self.train_env.query_expert(1 if mdp_mode else expert_ep_cnt)]
                )

                # observe reward and next obs (B=1, dim)
                if isinstance(action, tuple):
                    action = action[0]
                next_obs, reward, done, info, next_state = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # NOTE: designed by env
                term = self.config_env.terminal_fn(self.train_env, done_rollout, info)

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        state=ptu.get_numpy(state.squeeze(dim=0)),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                        next_state=ptu.get_numpy(next_state.squeeze(dim=0)),
                        is_expert=True,
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    state_list.append(state)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)
                    next_state_list.append(next_state)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()
                state = next_state.clone()
            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                if info["success"]:
                    self.policy_storage.add_episode(
                        observations=ptu.get_numpy(
                            torch.cat(obs_list, dim=0)
                        ),  # (L, dim)
                        states=ptu.get_numpy(torch.cat(state_list, dim=0)),  # (L, dim)
                        actions=ptu.get_numpy(act_buffer),  # (L, dim)
                        rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                        terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                        next_observations=ptu.get_numpy(
                            torch.cat(next_obs_list, dim=0)
                        ),  # (L, dim)
                        next_states=ptu.get_numpy(
                            torch.cat(next_state_list, dim=0)
                        ),  # (L, dim)
                    )
                    print(
                        f"Expert steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )
                    self._n_env_steps_total += steps
                    self._n_rollouts_total += 1
                    expert_ep_cnt += 1
            else:
                print(
                    f"Expert steps: {steps} term: {term} last_reward: {reward.item()}"
                )
                self._n_env_steps_total += steps
                self._n_rollouts_total += 1
                expert_ep_cnt += 1

        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0
            obs = ptu.from_numpy(self.train_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            state = ptu.from_numpy(self.train_env.get_state())  # state at reset
            state = state.reshape(1, state.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                (
                    obs_list,
                    act_list,
                    rew_list,
                    next_obs_list,
                    term_list,
                    state_list,
                    next_state_list,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        action, internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            state=state,
                            deterministic=False,
                        )
                    else:
                        action = self.agent.act(obs, state, deterministic=False)

                # observe reward and next obs (B=1, dim)
                if isinstance(action, tuple):
                    action = action[0]
                next_obs, reward, done, info, next_state = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # NOTE: designed by env
                term = self.config_env.terminal_fn(self.train_env, done_rollout, info)

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        state=ptu.get_numpy(state.squeeze(dim=0)),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                        next_state=ptu.get_numpy(next_state.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    state_list.append(state)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)
                    next_state_list.append(next_state)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()
                state = next_state.clone()
            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    states=ptu.get_numpy(torch.cat(state_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                    next_states=ptu.get_numpy(
                        torch.cat(next_state_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"Env steps: {self._n_env_steps_total + steps}/{self.n_env_steps_total} steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            aux_batch = self.sample_rl_batch(self.FLAGS.aux_batch_size)

            # Reconstructor update
            if "ours" in self.config_rl.algo:
                recon_loss = self.agent.update_recon(aux_batch)
            else:
                recon_loss = {}

            # History encoder update
            if "believer" in self.config_rl.algo:
                believer_loss = self.agent.update_believer_encoder(aux_batch)
            else:
                believer_loss = {}

            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.FLAGS.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            rl_losses.update(recon_loss)
            rl_losses.update(believer_loss)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, deterministic=True):
        self.agent.eval()  # set to eval mode for deterministic dropout

        returns_per_episode = np.zeros(self.config_env.eval_episodes)
        success_rate = np.zeros(self.config_env.eval_episodes)
        total_steps = np.zeros(self.config_env.eval_episodes)

        for task_idx in range(self.config_env.eval_episodes):
            step = 0
            running_reward = 0.0
            done_rollout = False

            obs = ptu.from_numpy(self.eval_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            state = ptu.from_numpy(self.eval_env.get_state())  # state at reset
            state = state.reshape(1, state.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

                # temporary storage
                (
                    obs_list,
                    act_list,
                    rew_list,
                    next_obs_list,
                    term_list,
                    state_list,
                    next_state_list,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                mask_list = []

            while not done_rollout:
                if self.agent_arch == AGENT_ARCHS.Memory:
                    action, internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        state=state,
                        deterministic=deterministic,
                    )
                else:
                    action = self.agent.act(obs, state, deterministic=deterministic)

                if isinstance(action, tuple):
                    action = action[0]
                # observe reward and next obs
                next_obs, reward, done, info, next_state = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )

                print_ins_reward = False
                if self.agent_arch == AGENT_ARCHS.Memory and print_ins_reward:
                    obs_list.append(obs)  # (1, dim)
                    state_list.append(state)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(done)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)
                    next_state_list.append(next_state)  # (1, dim)
                    mask_list.append(1 - done)

                    actions = torch.cat(act_list, dim=0)[:, None, :]
                    rewards = torch.cat(rew_list, dim=0)[:, None, :]
                    observs = torch.cat(obs_list, dim=0)[:, None, :]
                    next_observs = torch.cat(next_obs_list, dim=0)[:, None, :]
                    states = torch.cat(state_list, dim=0)[:, None, :]
                    next_states = torch.cat(next_state_list, dim=0)[:, None, :]
                    dones = torch.tensor(term_list).reshape(-1, 1)[:, None, :].cuda()
                    masks = torch.tensor(mask_list).reshape(-1, 1)[:, None, :].cuda()

                    states = torch.cat((states[[0]], next_states), dim=0)
                    observs = torch.cat((observs[[0]], next_observs), dim=0)
                    actions = torch.cat(
                        (ptu.zeros((1, 1, self.act_dim)).float(), actions), dim=0
                    )  # (T+1, B, dim)
                    rewards = torch.cat(
                        (ptu.zeros((1, 1, 1)).float(), rewards), dim=0
                    )  # (T+1, B, dim)
                    dones = torch.cat(
                        (ptu.zeros((1, 1, 1)).float(), dones), dim=0
                    )  # (T+1, B, dim)

                    ins_r = self.agent.critic.calc_intrinsic_rewards(
                        actions, rewards, observs, states, dones, masks
                    )
                    print(ptu.get_numpy(ins_r[-1][0][0]))

                # add raw reward
                running_reward += reward.item()
                step += 1
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: obs <- next_obs
                obs = next_obs.clone()
                state = next_state.clone()
            if print_ins_reward:
                print("Done")
            returns_per_episode[task_idx] = running_reward
            total_steps[task_idx] = step
            if "success" in info and info["success"] == True:  # keytodoor
                success_rate[task_idx] = 1.0

        self.agent.train()  # set it back to train
        return returns_per_episode, success_rate, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step("env_steps", self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular(k, v)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

    def log(self):
        logger.record_step("metrics/env_steps", self._n_env_steps_total)
        returns_eval, success_rate_eval, total_steps_eval = self.evaluate()
        logger.record_tabular("metrics/return", np.mean(returns_eval))
        logger.record_tabular("metrics/success", np.mean(success_rate_eval))
        logger.record_tabular("metrics/length", np.mean(total_steps_eval))
        logger.record_tabular(
            "time/time_cost", int(time.time() - self._start_time) / 3600.0
        )
        logger.record_tabular(
            "time/FPS",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        return np.mean(returns_eval)

    def save_model(self, total_steps, perf, to_cloud=False):
        fname = f"agent_{total_steps:0{self._digit()}d}_perf{perf:.3f}.pt"
        save_path = os.path.join(
            logger.get_dir(),
            fname,
        )
        torch.save(self.agent.state_dict(), save_path)

        if to_cloud:
            logger.save_to_cloud(save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        self.agent.eval()
        # torch.save(self.agent.critic.state_embedder.state_dict(), "s_encoder.pt")
        # torch.save(self.agent.critic.image_encoder.state_dict(), "s_encoder.pt")
        # breakpoint()
        print("load successfully from", ckpt_path)

    def _digit(self):
        # zero pad with total env steps
        return int(math.log10(self.n_env_steps_total) + 1)
