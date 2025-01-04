import numpy as np
import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu


def get_grad_norm(model):
    # mean of grad norm^2
    grad_norm = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_norm.append(p.grad.data.norm(2).item())
    if grad_norm:
        grad_norm = np.mean(grad_norm)
    else:
        grad_norm = 0.0
    return grad_norm


def env_step(env, action):
    # action: (A)
    # return: all 2D tensor shape (B=1, dim)
    action = ptu.get_numpy(action)
    if env.action_space.__class__.__name__ == "Discrete":
        action = np.argmax(action)  # one-hot to int
    next_obs, reward, done, info = env.step(action)
    next_state = env.get_state()

    # move to torch
    next_obs = ptu.from_numpy(next_obs).view(-1, next_obs.shape[0])
    reward = ptu.FloatTensor([reward]).view(-1, 1)
    done = ptu.from_numpy(np.array(done, dtype=int)).view(-1, 1)
    next_state = ptu.from_numpy(next_state).view(-1, next_state.shape[0])

    return next_obs, reward, done, info, next_state


class FeatureExtractor(nn.Module):
    """one-layer MLP with relu
    Used for extracting features for vector-based observations/actions/rewards

    NOTE: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch.linear is a linear transformation in the LAST dimension
    with weight of size (IN, OUT)
    which means it can support the input size larger than 2-dim, in the form
    of (*, IN), and then transform into (*, OUT) with same size (*)
    e.g. In the encoder, the input is (N, B, IN) where N=seq_len.
    """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return ptu.zeros(
                0,
            )  # useful for concat


class LinearSchedule:
    def __init__(self, init_value, end_value, transition_steps):
        self.value_from = init_value
        self.value_to = end_value
        self.nsteps = transition_steps

    def value(self, step) -> float:
        if step < 0:
            return self.value_from

        if step >= self.nsteps - 1:
            return self.value_to

        step_size = (self.value_to - self.value_from) / (self.nsteps)
        value = self.value_from + step * step_size
        return value