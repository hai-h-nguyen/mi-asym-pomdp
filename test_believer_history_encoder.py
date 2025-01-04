import torch
import torch.nn as nn
from policies.models.believer_encoders import BeliefVAEModel
from torchkit import pytorch_utils as ptu

belief_agg = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64)
)

belief_encoder = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64)
)

obs_space = {"image": (2, 1, 1)}
belief_vae = BeliefVAEModel(obs_space, 2, 2)

obs = torch.rand((51, 64, 2, 1, 1))
memory = torch.zeros(3, 64, belief_vae.semi_memory_size, device=ptu.device)

history_encoding, memory = belief_vae(obs, memory)
samples = belief_vae.sample(history_encoding)
belief_enc = belief_agg(
    torch.mean(belief_encoder(samples), dim=2)
)
print(belief_enc.shape)