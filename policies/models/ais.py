import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkit.networks import FlattenMlp
from torchkit import pytorch_utils as ptu


class LatentModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic latent transition models.
    E[o' | z, a] or E[z' | z, a], depends on num_obs
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(LatentModel, self).__init__()
        input_ndims = AIS_state_size + num_actions

        hidden_fcn = F.relu
        output_fcn = ptu.identity

        self.model = FlattenMlp(input_size=input_ndims,
                                output_size=num_obs,
                                hidden_sizes=(128, 128),
                                hidden_activation=hidden_fcn,
                                output_activation=output_fcn)

    def forward(self, x):
        return self.model(x)


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0)
