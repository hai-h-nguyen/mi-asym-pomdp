import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import FlattenMlp


class MLP(nn.Module):
    name = "mlp"
    rnn_class = FlattenMlp

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.model = self.rnn_class(
            input_size=input_size,
            hidden_sizes=(128, 128),
            output_size=hidden_size,
        )
        self.hidden_size = hidden_size

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (num_layers=1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (num_layers=1, B, hidden_size), only used in inference
        """
        output = self.model(inputs)
        return output, h_0

    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((1, batch_size, self.hidden_size)).float()
