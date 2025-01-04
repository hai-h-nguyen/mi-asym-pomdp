import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkit.networks import FlattenMlp
import os
import wandb
from policies.models.models_cv_mim import *

class ReprModel(nn.Module):

    def __init__(self, obs_dim, state_dim, action_dim, config):
        super().__init__()

        self.latent_dim = config.latent_dim

        self.obs_dim = obs_dim
        self.state_dim = state_dim

        dynamics_input_shape = 2*self.latent_dim + 1

        self.dynamics_model = FlattenMlp(input_size=dynamics_input_shape,
                                         output_size=config.d_output_size,
                                         hidden_sizes=config.d_hidden_size)

        self.next_state_model = nn.Linear(config.d_output_size, self.latent_dim)
        self.next_obs_model = nn.Linear(config.d_output_size, self.latent_dim)
        self.reward_model = nn.Linear(config.d_output_size, 1)

        self.state_encoder = FlattenMlp(input_size=self.state_dim,
                                        output_size=2*self.latent_dim,
                                        hidden_sizes=config.s_hidden_size)

        self.obs_encoder = FlattenMlp(input_size=self.obs_dim,
                                      output_size=2*self.latent_dim,
                                      hidden_sizes=config.o_hidden_size)
        # self.state_encoder = SimpleModel(self.state_dim, 2*self.latent_dim,
        #                                  hidden_dim=128, num_residual_linear_block=1, num_layers_per_block=1,
        #                                  dropout_rate=0.0, use_batch_norm=False)
        
        # self.obs_encoder = SimpleModel(self.obs_dim, 2*self.latent_dim,
        #                                hidden_dim=128, num_residual_linear_block=1, num_layers_per_block=1,
        #                                dropout_rate=0.0, use_batch_norm=False)
        
        print('[INFO] Number of parameters in state encoder:', sum(p.numel() for p in self.state_encoder.parameters() if p.requires_grad))
        print('[INFO] Number of parameters in obs encoder:', sum(p.numel() for p in self.obs_encoder.parameters() if p.requires_grad))
        # exit()
    def encode_state(self, state):
        output = self.state_encoder(state)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        return encoder_mean, encoder_std

    def encode_obs(self, obs):
        output = self.obs_encoder(obs)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        return encoder_mean, encoder_std

    def predict_next(self, state, obs, action):
        encoder_mean_s, encoder_std_s = self.encode_state(state)
        zs = encoder_mean_s + torch.randn_like(encoder_mean_s) * encoder_std_s

        encoder_mean_o, encoder_std_o = self.encode_obs(obs)
        zo = encoder_mean_o + torch.randn_like(encoder_mean_o) * encoder_std_o

        if len(action.shape) == 1:
            action = action.float().unsqueeze(dim=1)
        else:
            action = action.float()
        embedding = self.dynamics_model(torch.cat([zs, zo, action], dim=1))
        return self.next_state_model(embedding), \
            self.next_obs_model(embedding), \
            self.reward_model(embedding)

    def save(self, path, to_cloud=False):
        torch.save(self.state_encoder.state_dict(),
                   os.path.join(path, "s_encoder.pt"))
        torch.save(self.obs_encoder.state_dict(),
                   os.path.join(path, "o_encoder.pt"))
        # torch.save(self.state_dict(),
        #            os.path.join(path, "repr_model.pt"))

        if to_cloud:
            wandb.save(os.path.join(path, "s_encoder.pt"))
            wandb.save(os.path.join(path, "o_encoder.pt"))