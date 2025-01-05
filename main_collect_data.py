
import torch
from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm

from utils.augmentation import perturb_1, get_random_transform_params
from envs.make_env import make_env
import numpy as np
from matplotlib import pyplot as plt


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_env"])

# training settings
flags.DEFINE_integer("seed", 42, "Random seed (default 42).")
flags.DEFINE_integer("episodes", 3000, "Number of episodes during training.")
flags.DEFINE_integer("random_episodes", 0, "Number of random episodes.")
flags.DEFINE_boolean("aug", False, "Augmentations.")
flags.DEFINE_integer("n_aug", 1, "Number of augmentations.")

def main(argv):
    seed = FLAGS.seed
    num_episodes = FLAGS.episodes
    random_episodes = FLAGS.random_episodes
    planner_episodes = num_episodes - random_episodes
    aug = FLAGS.aug
    assert aug == False
    n_aug = FLAGS.n_aug
    config_env = FLAGS.config_env

    config_env, env_name = config_env.create_fn(config_env)
    env = make_env(env_name, seed)
    max_episode_length = env.max_episode_steps
    obs_dim = env.observation_space.shape[0]
    state_dim = env.state_space.shape[0]

    try:
        action_dim = env.action_space.shape[0]
    except IndexError:
        action_dim = -1

    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")

    if aug and planner_episodes > 0:
        planner_episodes  = planner_episodes * (n_aug + 1)
    obss = torch.zeros((random_episodes + planner_episodes, max_episode_length, obs_dim))
    states = torch.zeros((random_episodes + planner_episodes, max_episode_length, state_dim))
    rewards = torch.zeros((random_episodes + planner_episodes, max_episode_length))

    if action_dim == -1:
        actions = torch.zeros((random_episodes + planner_episodes, max_episode_length))
    else:
        actions = torch.zeros((random_episodes + planner_episodes, max_episode_length, action_dim))
    masks = torch.zeros((random_episodes + planner_episodes, max_episode_length))

    total_steps = 0
    u = 0.0

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()

        state = env.get_state()
        done = False
        step = 0
        returnn = 0

        while not done and step < max_episode_length:
            obss[episode, step] = torch.tensor(obs)
            states[episode, step] = torch.tensor(state)
            if episode < random_episodes:
                action = env.action_space.sample()
            else:
                action = env.query_expert(episode)
            actions[episode, step] = torch.tensor(action)
            masks[episode, step] = 1 - done # The first 0 is where we no longer have data

            obs, reward, done, _ = env.step(action)
            assert obs.max() <= 1.0 and obs.min() >= -1.0
            state = env.get_state()
            assert state.max() <= 1.0 and state.min() >= -1.0
            
            # assert reward == 1 or reward == 0
            if reward > 0:
                u += 1
            rewards[episode, step] = reward
            returnn += reward

            step += 1
            total_steps += 1

        if aug and episode >= random_episodes and planner_episodes > 0:
            print("oooo")
            for ep_idx in range(n_aug):
                theta, trans, pivot = get_random_transform_params(84)
                obss[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1))] = obss[episode].clone()
                states[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1))] = states[episode].clone()
                actions[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1))] = actions[episode].clone()
                masks[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1))] = masks[episode].clone()
                rewards[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1))] = rewards[episode].clone()
                for aug_step in range(step):
                    new_obs = obss[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step].clone()
                    new_obs = new_obs.cpu().numpy()
                    new_obs = np.reshape(new_obs, (2, 84, 84))
                    new_state = states[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step].clone()
                    new_state = new_state.cpu().numpy()
                    new_state = np.reshape(new_state, (2, 84, 84))
                    new_action = actions[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step].clone()
                    new_action = new_action.cpu().numpy() 
                    
                    aug_obs, aug_state, dxy = perturb_1(new_obs,
                                                        new_state,
                                                        new_action[1:3],
                                                        theta, trans, pivot,
                                                        set_trans_zero=True)

                    new_obs = aug_obs
                    new_state = aug_state
                    new_action[1:3] = dxy
                   
                    obss[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step] = torch.tensor(new_obs.flatten())
                    states[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step] = torch.tensor(new_state.flatten())
                    actions[episode + int(((ep_idx + 1) * planner_episodes) / (n_aug + 1)), aug_step] = torch.tensor(new_action)
    print("Total Steps: ", total_steps)
    print("Success Rate: ", u / num_episodes)

    # Save pytorch tensor
    if aug:
        torch.save({"obss": obss,
                    "states": states,
                    "actions": actions,
                    "masks": masks,
                    "rewards": rewards,
                    }, f'{env_name}_s{seed}_{num_episodes}ep_aug_{n_aug}.data')
    else:
        torch.save({"obss": obss,
                    "states": states,
                    "actions": actions,
                    "masks": masks,
                    "rewards": rewards,
                    }, f'{env_name}_s{seed}_{num_episodes}ep.data')


if __name__ == "__main__":
    app.run(main)