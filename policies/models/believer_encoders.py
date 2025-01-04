import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchkit import pytorch_utils as ptu
from policies.seq_models import SEQ_MODELS


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Image3dEncoder(nn.Module):
    def __init__(self, env_name):
        super().__init__()

        if env_name in ["CarFlag-2D-v0"]:
            self.embedding_size = 128
            self.image_size = (2, 11, 11)
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1296, self.embedding_size)
            )
        elif env_name in ["Sphinx-v0"]:
            self.embedding_size = 128
            self.image_size = (2, 6, 6)
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256, self.embedding_size)
            )

        elif env_name in ["BlockPulling-v0", "BlockPushing-v0", "DrawerOpening-v0"]:
            self.embedding_size = 128
            self.image_size = (2, 84, 84)
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(576, self.embedding_size)
            )

    def forward(self, obs):
        batch = obs.shape[1]
        x = self.image_conv(obs.reshape(-1, *self.image_size))
        x = x.reshape((-1, batch, self.embedding_size))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        input_size = obs_space['image'][0] * obs_space['image'][1] * obs_space['image'][2]

        self.embedding_size = 128  # image_embedding size

        self.image_conv = nn.Sequential(
            nn.Linear(input_size, self.embedding_size),
        )

    def forward(self, obs):
        if torch.is_tensor(obs):
            x = self.image_conv(obs.reshape(obs.shape[0], obs.shape[1], -1))
        else:
            x = self.image_conv(obs.image[:, :, :, :].reshape(obs.image.shape[0], -1))

        return x


class HistoryEncoder(nn.Module):
    def __init__(self, env_name, obs_space):
        super().__init__()

        if env_name in ["HeavenHell-v0"]:
            self.image_conv = ImageEncoder(obs_space)
        else:
            self.image_conv = Image3dEncoder(env_name)

        self.image_embedding_size = self.image_conv.embedding_size

        self.memory_rnn = nn.GRU(self.image_embedding_size,
                                 self.semi_memory_size,
                                 num_layers=3)

        self.embedding_size = self.semi_memory_size

        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_size + self.image_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        )

    @property
    def memory_size(self):
        return 3*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return 256

    def forward(self, obs, memory):
        single_step = False
        x = self.image_conv(obs)
        if len(x.shape) == 2:
            single_step = True
            x = x.unsqueeze(1)
            memory = memory[:, None, :].reshape(3, -1, self.semi_memory_size)
        output, memory = self.memory_rnn(x, memory)
        prediction = self.predictor(torch.cat((output, x), dim=-1))
        if single_step:
            prediction = prediction[:, 0, :]
            memory = memory.reshape(-1, self.memory_size)

        return prediction, memory


class HistoryEncoderGPT(nn.Module):
    def __init__(self, env_name, obs_space, config_seq):
        super().__init__()

        if env_name in ["HeavenHell-v0"]:
            self.image_conv = ImageEncoder(obs_space)
        else:
            self.image_conv = Image3dEncoder(env_name)

        self.image_embedding_size = 128
        config_seq.seq_model_config.hidden_size = self.image_embedding_size

        self.memory_rnn = SEQ_MODELS[config_seq.seq_model_config.name](
            input_size=self.image_embedding_size,
            **config_seq.seq_model_config.to_dict()
        )

        self.predictor = nn.Sequential(
            nn.Linear(2*self.image_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    @property
    def memory_size(self):
        return 3*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return 256

    def forward(self, obs, memory):
        single_step = False
        x = self.image_conv(obs)
        if len(x.shape) == 2:
            single_step = True
            x = x.unsqueeze(1)
            memory = memory[:, None, :].reshape(3, -1, self.semi_memory_size)
        output, memory = self.memory_rnn(x, memory)
        prediction = self.predictor(torch.cat((output, x), dim=-1))
        if single_step:
            prediction = prediction[:, 0, :]
            memory = memory.reshape(-1, self.memory_size)

        return prediction, memory


class BeliefVAEModel(nn.Module):
    def __init__(self, obs_space, env_name, x_dim, x_size,
                 latent_dim=8):
        super().__init__()

        self.x_dim = x_dim
        self.x_size = x_size
        self.latent_dim = latent_dim
        print(x_dim, "xdim")
        print(x_size, "num pixels")
        print("latent dim of VAE is", latent_dim)

        self.history_model = HistoryEncoder(env_name, obs_space)
        self.context_dim = self.history_model.semi_memory_size
        state_features_dim = x_size

        # Outputs a mean and variance for a diagonal Gaussian in latent space
        self.vae_encoder = nn.Sequential(
            nn.Linear(state_features_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_dim)
        ).to(ptu.device)

        self.vae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.x_dim)
        ).to(ptu.device)

    def get_zero_internal_state(self, batch_size=1):
        return torch.zeros(3, batch_size, self.history_model.semi_memory_size, device=ptu.device)

    @property
    def memory_size(self):
        return self.history_model.memory_size

    @property
    def semi_memory_size(self):
        return self.history_model.semi_memory_size

    def forward(self, obs, memory):
        encoding, memory = self.history_model(obs, memory)
        return encoding, memory

    def encoder_dist(self, x, context):
        out = self.vae_encoder(torch.cat([x, context], dim=-1))
        mean = out[:, :, :self.latent_dim]
        std = F.softplus(out[:, :, self.latent_dim:], threshold=1) + 1e-1

        return mean, std

    def decoder_dist(self, z, context):
        out = self.vae_decoder(torch.cat([z.to(ptu.device), context.to(ptu.device)], dim=-1))
        mean = out[:, :, self.x_dim:]
        std = F.softplus(out[:, :, : self.x_dim], threshold=1) + 1e-1
        return mean, std

    def sample(self, context):
        bs = context.shape[0]
        length = context.shape[1]
        zs = torch.randn(bs * 30, length, self.latent_dim)
        mean, std = self.decoder_dist(zs, context.repeat_interleave(30, dim=0))
        samples = Normal(mean, std).sample().reshape(bs, length, 30, -1)

        return samples


class BeliefVAEModelGPT(nn.Module):
    def __init__(self, obs_space, env_name, x_dim, x_size,
                 config_seq,
                 latent_dim=8):
        super().__init__()

        self.x_dim = x_dim
        self.x_size = x_size
        self.latent_dim = latent_dim
        print(x_dim, "xdim")
        print(x_size, "num pixels")
        print("latent dim of VAE is", latent_dim)

        self.history_model = HistoryEncoderGPT(env_name, obs_space, config_seq)
        self.context_dim = self.history_model.semi_memory_size
        state_features_dim = x_size

        # Outputs a mean and variance for a diagonal Gaussian in latent space
        self.vae_encoder = nn.Sequential(
            nn.Linear(state_features_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_dim)
        ).to(ptu.device)

        self.vae_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.x_dim)
        ).to(ptu.device)

    def get_zero_internal_state(self, batch_size=1):
        if batch_size == 1:
            return self.history_model.memory_rnn.get_zero_internal_state()
        else:
            return self.history_model.memory_rnn.get_zero_internal_state(batch_size)

    def forward(self, obs, memory):
        encoding, memory = self.history_model(obs, memory)
        return encoding, memory

    def encoder_dist(self, x, context):
        out = self.vae_encoder(torch.cat([x, context], dim=-1))
        mean = out[:, :, :self.latent_dim]
        std = F.softplus(out[:, :, self.latent_dim:], threshold=1) + 1e-1

        return mean, std

    def decoder_dist(self, z, context):
        out = self.vae_decoder(torch.cat([z.to(ptu.device), context.to(ptu.device)], dim=-1))
        mean = out[:, :, self.x_dim:]
        std = F.softplus(out[:, :, : self.x_dim], threshold=1) + 1e-1
        return mean, std

    def sample(self, context):
        bs = context.shape[0]
        length = context.shape[1]
        zs = torch.randn(bs * 30, length, self.latent_dim)
        mean, std = self.decoder_dist(zs, context.repeat_interleave(30, dim=0))
        samples = Normal(mean, std).sample().reshape(bs, length, 30, -1)

        return samples

class RepresentationModel(nn.Module):
    def __init__(self, env_name, state_space, latent_dim=16):
        super().__init__()

        self.latent_dim = latent_dim
        print(latent_dim, "Representation Model Latent Dim")

        self.use_cnn = False
        if env_name in ["HeavenHell-v0"]:
            n = state_space["image"][0]
            m = state_space["image"][1]
            k = state_space["image"][2]
            self.image_embedding_size = n * m * k
        else:
            self.use_cnn = True
            if env_name in ["CarFlag-2D-v0"]:
                self.image_embedding_size = 128
                self.image_size = (2, 11, 11)

                self.state_image_cnn = nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(1296, self.image_embedding_size)
                )
            elif env_name in ["Sphinx-v0"]:
                self.image_embedding_size = 128
                self.image_size = (2, 6, 6)

                self.state_image_cnn = nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(256, self.image_embedding_size)
                )
            elif env_name in ["BlockPulling-v0", "BlockPushing-v0", "DrawerOpening-v0"]:
                self.image_embedding_size = 128
                self.image_size = (2, 84, 84)
                self.state_image_cnn = nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(576, self.image_embedding_size)
                )

        state_input_size = self.image_embedding_size
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*latent_dim)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def encode_state(self, state):
        if self.use_cnn:
            batch = state.shape[1]
            state = self.state_image_cnn(state.reshape(-1, *self.image_size))
            state = state.reshape((-1, batch, self.image_embedding_size))
        output = self.state_encoder(state)
        encoder_mean = output[:, :, :self.latent_dim]
        encoder_std = F.softplus(output[:, :, self.latent_dim:], threshold=1) + 1e-5
        return encoder_mean, encoder_std
