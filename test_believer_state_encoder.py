import torch
from policies.models.believer_encoders import RepresentationModel

state_space = {"image": (2, 1, 1)}
belief_vae = RepresentationModel(state_space, False)

state = torch.rand((51, 64, 2, 1, 1))
o = belief_vae.encode_state(state)
