import torch
import torch.nn as nn
from policies.models.believer_encoders import BeliefVAEModelGPT
from torchkit import pytorch_utils as ptu
import argparse
from ml_collections import ConfigDict


parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default="BlockPulling-v0")

args = parser.parse_args()

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

config = ConfigDict()
config.model = ConfigDict()
# seq_model_config specific
config.model.seq_model_config = ConfigDict()
config.model.seq_model_config.name = "gpt"

config.model.seq_model_config.hidden_size = (
    128  # NOTE: will be overwritten by name_fn
)
config.model.seq_model_config.n_layer = 1
config.model.seq_model_config.n_head = 2
config.model.seq_model_config.pdrop = 0.1
config.model.seq_model_config.position_encoding = "sine"
config.model.seq_model_config.max_seq_length = 51

obs_space = {"image": (2, 84, 84)}
belief_vae = BeliefVAEModelGPT(obs_space, args.domain,
                               2, 2, config)

obs = torch.rand((51, 64, 2, 84, 84))
# memory = belief_vae.history_model.memory_rnn.get_zero_internal_state()
memory = None

history_encoding, memory = belief_vae(obs, memory)
samples = belief_vae.sample(history_encoding)
belief_enc = belief_agg(
    torch.mean(belief_encoder(samples), dim=2)
)
print(belief_enc.shape)
