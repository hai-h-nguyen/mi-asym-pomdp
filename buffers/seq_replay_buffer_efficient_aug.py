from .seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer
from utils.augmentation import perturb, get_random_transform_params
import matplotlib.pyplot as plt


class RAMEfficient_SeqReplayBufferAug(RAMEfficient_SeqReplayBuffer):
    buffer_type = "seq_efficient_rot"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        state_dim,
        action_dim,
        sampled_seq_len: int,
        observation_type,
        num_augs: int,
        sample_weight_baseline: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            max_replay_buffer_size,
            observation_dim,
            state_dim,
            action_dim,
            sampled_seq_len,
            observation_type,
            sample_weight_baseline,
            **kwargs,
        )
        self.num_aug = num_augs
        self._save_image = False
        if action_dim == 4:
            self.start_dx = 0
        elif action_dim == 5:
            self.start_dx = 1
        else:
            raise ValueError("Invalid action_dim")
        self.flatten_dim = 2 * 84 * 84

    def add_episode(self, observations, actions, rewards,
                    terminals, next_observations, states, next_states):
        """
        NOTE: must add one whole episode/sequence/trajectory,
                        not some partial transitions
        the length of different episode can vary, but must be greater than 2
                so that the end of valid_starts is 0.

        all the inputs have 2D shape of (L, dim)
        """
        super().add_episode(observations, actions, rewards, terminals,
                            next_observations, states, next_states)

        self._aug_and_add(observations, actions, states, rewards,
                          terminals, next_observations, next_states)

    def _aug_and_add(self, observations, actions, states, rewards,
                     terminals, next_observations, next_states):
        """
        Augment an episode and add it to the replay buffer
        """
        seq_len = observations.shape[0]

        reshaped_obss = observations.reshape(seq_len, 2, 84, 84)
        reshaped_states = states.reshape(seq_len, 2, 84, 84)
        reshaped_n_obss = next_observations.reshape(seq_len, 2, 84, 84)
        reshaped_n_states = next_states.reshape(seq_len, 2, 84, 84)

        for _ in range(self.num_aug):
            new_observations = reshaped_obss.copy()
            new_states = reshaped_states.copy()
            new_actions = actions.copy()
            new_next_observations = reshaped_n_obss.copy()
            new_next_states = reshaped_n_states.copy()

            # Compute random rigid transform.
            # Same for the entire history
            theta, trans, pivot = get_random_transform_params(84)

            for idx in range(seq_len):
                _obss, _next_obss, _states, _next_states, _dxy = perturb(reshaped_obss[idx].copy(),
                                                                         reshaped_n_obss[idx].copy(),
                                                                         reshaped_states[idx].copy(),
                                                                         reshaped_n_states[idx].copy(),
                                                                         actions[idx][self.start_dx:self.start_dx + 2],
                                                                         theta, trans, pivot,
                                                                         set_trans_zero=True)

                new_observations[idx] = _obss
                new_states[idx] = _states
                new_actions[idx][self.start_dx:self.start_dx + 2] = _dxy
                new_next_observations[idx] = _next_obss
                new_next_states[idx] = _next_states

                if self._save_image:
                    plt.imshow(reshaped_obss[idx][0])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"obs_{idx}_before_c0.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(reshaped_n_obss[idx][1])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"obs_{idx}_before_c1.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(reshaped_states[idx][0])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"state_{idx}_before_c0.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(reshaped_states[idx][1])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"state_{idx}_before_c1.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(new_observations[idx][0])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"obs_{idx}_after_c0.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(new_observations[idx][1])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"obs_{idx}_after_c1.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(new_states[idx][0])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"state_{idx}_after_c0.png", bbox_inches='tight')
                    plt.close()

                    plt.imshow(new_states[idx][1])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"state_{idx}_after_c1.png", bbox_inches='tight')
                    plt.close()

            super().add_episode(
                new_observations.reshape(seq_len, self.flatten_dim),
                new_actions,
                rewards,
                terminals,
                new_next_observations.reshape(seq_len, self.flatten_dim),
                new_states.reshape(seq_len, self.flatten_dim),
                new_next_states.reshape(seq_len, self.flatten_dim),
            )
