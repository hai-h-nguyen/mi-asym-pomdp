import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkit.networks import FlattenMlp
from torchkit import pytorch_utils as ptu


tanh_name = "tanh"
sigmoid_name = "sigmoid"
ident_name = "ident"
relu_name = "relu"
ACTIVATIONS = {
    tanh_name: torch.tanh,
    relu_name: F.relu,
    sigmoid_name: torch.sigmoid,
    ident_name: ptu.identity,
}

# Reconstruction losses
mse_loss_name = "mse"
cosine_loss_name = "cosine"
l1_loss_name = "l1"
RECON_LOSS_FCNS = {
    mse_loss_name: nn.MSELoss(reduction="none"),
    cosine_loss_name: nn.CosineSimilarity(dim=-1, eps=1e-6),
    l1_loss_name: nn.L1Loss(reduction="none"),
}


class State_Reconstructor(nn.Module):
    """A class to reconstruct state-related information (s or phi(s))
    from history representations (\bar{h}_t)
    Args:
        nn (_type_): _description_
    """
    def __init__(self, config, output_dim, state_dim, input_size):
        super().__init__()
        self.recon_type = config.type

        output_size = 0
        if self.recon_type == "recon_s":  # reconstruct full s
            output_size = state_dim
        elif self.recon_type == "recon_phi_s":  # reconstruct phi(s)
            output_size = output_dim
        else:
            output_size = 0
        self.output_size = output_size

        hidden_fcn = ACTIVATIONS[config.hidden_act_fcn]
        output_fcn = ACTIVATIONS[config.output_act_fcn]

        self.output_size = output_size
        self.reconstructor = FlattenMlp(input_size=input_size,
                                        output_size=output_size,
                                        hidden_sizes=config.hidden_dims,
                                        hidden_activation=hidden_fcn,
                                        output_activation=output_fcn)

    def forward(self, hidden_states):
        """_summary_

        Args:
            hidden_states (_type_): _description_

        Returns:
            _type_: reconstructed state or state features
        """
        return self.reconstructor(hidden_states)
