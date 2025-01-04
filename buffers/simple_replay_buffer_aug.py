from .simple_replay_buffer import SimpleReplayBuffer
from utils.augmentation import perturb, get_random_transform_params
import matplotlib.pyplot as plt


class SimpleReplayBufferAug(SimpleReplayBuffer):
    buffer_type = "markov_rot"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        state_dim,
        max_trajectory_len: int,
        num_augs: int,
        add_timeout: bool = False,
        **kwargs
    ):
        super().__init__(
            max_replay_buffer_size,
            observation_dim,
            action_dim,
            state_dim,
            max_trajectory_len,
            add_timeout,
            **kwargs
        )

        self._num_augs = num_augs
        self._save_image = False
        if action_dim == 4:
            self.start_dx = 0
        elif action_dim == 5:
            self.start_dx = 1
        else:
            raise ValueError("Invalid action_dim")

    def add_sample(
        self,
        observation,
        action,
        state,
        reward,
        terminal,
        next_observation,
        next_state,
        timeout=None,
        is_expert=False,
        **kwargs
    ):
        super().add_sample(
            observation,
            action,
            state,
            reward,
            terminal,
            next_observation,
            next_state,
            timeout,
            is_expert,
        )
        self._aug_and_add(
            observation,
            action,
            state,
            reward,
            terminal,
            next_observation,
            next_state,
            timeout,
            is_expert,
        )

    def _aug_and_add(
        self,
        observation,
        action,
        state,
        reward,
        terminal,
        next_observation,
        next_state,
        timeout,
        is_expert
    ):

        reshaped_obs = observation.reshape(2, 84, 84)
        reshaped_state = state.reshape(2, 84, 84)
        reshaped_n_obs = next_observation.reshape(2, 84, 84)
        reshaped_n_state = next_state.reshape(2, 84, 84)

        for aug_idx in range(self._num_augs):
            new_obs = reshaped_obs.copy()
            new_state = reshaped_state.copy()
            new_action = action.copy()
            new_n_obs = reshaped_n_obs.copy()
            new_n_state = reshaped_n_state.copy()

            theta, trans, pivot = get_random_transform_params(84)

            if self._save_image:
                plt.imshow(reshaped_obs[0])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"obs_{aug_idx}_before_c0.png", bbox_inches='tight')
                plt.close()

                plt.imshow(reshaped_obs[1])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"obs_{aug_idx}_before_c1.png", bbox_inches='tight')
                plt.close()

                plt.imshow(reshaped_state[0])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"state_{aug_idx}_before_c0.png", bbox_inches='tight')
                plt.close()

                plt.imshow(reshaped_state[1])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"state_{aug_idx}_before_c1.png", bbox_inches='tight')
                plt.close()

            _obs, _next_obs, _state, _next_state, _dxy = perturb(reshaped_obs.copy(),
                                                                 reshaped_n_obs.copy(),
                                                                 reshaped_state.copy(),
                                                                 reshaped_n_state.copy(),
                                                                 action[self.start_dx:self.start_dx + 2],
                                                                 theta, trans, pivot,
                                                                 set_trans_zero=True)

            new_obs = _obs
            new_state = _state
            new_action[self.start_dx:self.start_dx + 2] = _dxy
            new_n_obs = _next_obs
            new_n_state = _next_state

            if self._save_image:
                plt.imshow(new_obs[0])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"obs_{aug_idx}_after_c0.png", bbox_inches='tight')
                plt.close()

                plt.imshow(new_obs[1])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"obs_{aug_idx}_after_c1.png", bbox_inches='tight')
                plt.close()

                plt.imshow(new_state[0])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"state_{aug_idx}_after_c0.png", bbox_inches='tight')
                plt.close()

                plt.imshow(new_state[1])
                plt.clim(0, 0.3)
                plt.colorbar()
                plt.savefig(f"state_{aug_idx}_after_c1.png", bbox_inches='tight')
                plt.close()

            super().add_sample(
                new_obs.flatten(),
                new_action,
                new_state.flatten(),
                reward,
                terminal,
                new_n_obs.flatten(),
                new_n_state.flatten(),
                timeout,
                is_expert,
            )
