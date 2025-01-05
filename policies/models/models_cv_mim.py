import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchkit.pytorch_utils import *
from policies.models.mi_estimators import CLUBSample

### MODEL DEFINITION ###
def build_linear(input_dim, output_dim):
    """Builds a linear layer.

    Args:
      hidden_dim: An integer for the dimension of the output.
      weight_max_norm: A float for the maximum weight norm to clip at. Use
        non-positive to ignore.
      weight_initializer: A string for kernel weight initializer.
      name: A string for the name of the layer.
      **_: Unused bindings for keyword arguments.

    Returns:
      A configured linear layer.
    """
    linear_layer = nn.Linear(input_dim, output_dim)
    fanin_init(linear_layer.weight)
    linear_layer.bias.data.fill_(0.1)
    return linear_layer

def build_linear_layers(input_dim, hidden_dim, num_layers, dropout_rate=0.0, use_batch_norm=False):
    """
    Builds linear layers.
    """
    linear_layers = nn.Sequential()
    for _ in range(num_layers):
        linear_layers.append(build_linear(input_dim, hidden_dim))
        if use_batch_norm:
            linear_layers.append(nn.BatchNorm1d(hidden_dim))
        linear_layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            linear_layers.append(nn.Dropout(dropout_rate))
        input_dim = hidden_dim
    return linear_layers

class PointEmbedder(nn.Module):
    """
    Point Embedder
    """
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedder = embedding_layer
    def forward(self, inputs):
        return self.embedder(inputs)

class SimpleModel(nn.Module):
    """
    Simple model architecture with a point
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=1024,
                 num_residual_linear_block=2,
                 num_layers_per_block=2,
                 from_flattened=False,
                 **kwargs):
        """Initializer.

        Args:
          input_dim: An integer for the input dimension.
          output_dim: A integer for the shape of the output.
          hidden_dim: An integer for the dimension of linear layers.
          num_residual_linear_blocks: An integer for the number of residual linear blocks.
          num_layers_per_block: An integer for the number of layers in each block.
          **kwargs: A dictionary for additional arguments. Supported arguments
            include `weight_max_norm`, `weight_initializer`, `dropout_rate` and
            `use_batch_norm`.
        """
        super().__init__()
        self.embedding_size = output_dim
        self.from_flattened = from_flattened
        self.blocks = nn.ModuleList()
        self.blocks.add_module(name="fc0", module=build_linear_layers(input_dim, hidden_dim, 1, **kwargs))
        for i in range(num_residual_linear_block):
            self.blocks.add_module(name="fc{}".format(i+1), module=build_linear_layers(hidden_dim, hidden_dim, num_layers_per_block, **kwargs))
        self.blocks.add_module(name="embedder", module=PointEmbedder(embedding_layer=build_linear(hidden_dim, output_dim)))

    def forward(self, inputs):
        activations = {}
        x = inputs
        for name, block in self.blocks.named_children():
            x = block(x)
            activations[name] = x
        output = activations['embedder']
        if self.from_flattened:
            return output
        else:
            return output, activations

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w

def build_conv_block(h_w, in_channels, out_channels, kernel_size, stride, pool=False, pool_size=0):
    conv_layers = nn.Sequential()
    conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    conv_layers.append(nn.ReLU())
    output_size = conv_output_shape(h_w, kernel_size=kernel_size, stride=stride)
    if pool:
        assert pool_size > 0
        conv_layers.append(nn.MaxPool2d((pool_size, pool_size)))
        output_size = (output_size[0] // pool_size, output_size[1] // pool_size)
    return conv_layers, output_size

class SimpleModelCNN(nn.Module):
    """
    Simple model architecture with a point
    """
    def __init__(self,
                 img_shape,
                 output_dim=32,
                 channels=[64, 128, 256],
                 kernel_size=[2, 2, 2],
                 strides=[1, 1, 1],
                 pool=[True, True, False],
                 pool_size=[2, 2, 0],
                 from_flattened=False,
                 ):
        super().__init__()
        assert len(channels) == len(kernel_size) == len(strides) == len(pool) == len(pool_size), "All lists must be the same length"
        self.shape = img_shape
        self.from_flattened = from_flattened
        self.embedding_size = output_dim
        self.channels = [img_shape[0]] + list(channels)
        self.blocks = nn.ModuleList()
        h_w = self.shape[-2:]
        for i in range(len(self.channels)-1):
            conv_layers, h_w = build_conv_block(h_w, self.channels[i], self.channels[i+1], kernel_size[i], strides[i], pool=pool[i], pool_size=pool_size[i])
            self.blocks.add_module(name="conv{}".format(i), module=conv_layers)
        hidden_dim = h_w[0] * h_w[1] * self.channels[-1]
        self.blocks.add_module(name="embedder", module=PointEmbedder(embedding_layer=build_linear(hidden_dim, output_dim))) 
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        activations = {}
        if self.from_flattened:
            # inputs size (T, B, C*H*W)
            assert inputs.ndim == 3
            l, b, d = inputs.shape
            inputs = inputs.reshape(l*b, self.shape[0], self.shape[1], self.shape[2])
        x = inputs
        for name, block in self.blocks.named_children():
            if name == 'embedder':
                x = x.contiguous().view(inputs.shape[0], -1)
            x = block(x)
            activations[name] = x
        output = activations['embedder']
        if not self.from_flattened:
            return output, activations
        else:
            return output.reshape(l, b, self.embedding_size)

### LOSS DEFINITION ###
def compute_positive_expectation(samples, measure, reduce_mean=False):
    """Computes the positive part of a divergence or difference.

    Args:
      samples: A tensor for the positive samples.
      measure: A string for measure to compute. Now support JSD
      reduce_mean: A boolean indicating whether to reduce results

    Returns:
      A tensor (has the same shape as samples) or scalar (if reduced) for the positive expectation of the inputs.
    """
    if measure == "JSD":
        expectation = math.log(2.) - F.softplus(-samples)
    elif measure == "W1":
        expectation = samples
    else:
        raise ValueError
    if reduce_mean:
        expectation = torch.mean(expectation)
    return expectation

def compute_negative_expectation(samples, measure, reduce_mean=False):
    """Computes the negative part of a divergence or difference.

    Args:
      logits: A tensor for the logits.
      measure: A string for measure to compute. Now support JSD
      reduce_mean: A boolean indicating whether to reduce results

    Returns:
      A tensor (has the same shape as logits) or scalar (if reduced) for the negative expectation of the inputs.
    """
    if measure == "JSD":
        expectation = F.softplus(-samples) + samples - math.log(2.)
    elif measure == "W1":
        expectation = samples
    else:
        raise ValueError
    if reduce_mean:
        expectation = torch.mean(expectation)
    return expectation

def compute_fenchel_dual_loss(local_features, global_features, measure, positive_indicator_matrix=None):
    """Computes the f-divergence loss.

    It is the distance between positive and negative joint distributions.
    Divergences (measures) supported are Jensen-Shannon (JSD),
    GAN (equivalent to JSD), Squared Hellinger (H2), KL and reverse KL (RKL).

    Reference:
      Hjelm et al. Learning deep representations by mutual information estimation
      and maximization. https://arxiv.org/pdf/1808.06670.pdf.

    Args:
      local_features: A tensor for local features. [batch_size, num_locals, feature_dim].
      global_features: A tensor for local features. [batch_size, num_globals, feature_dim].
      measure: A string for f-divergence measure.
      positive_indicator_matrix: A tensor for indicating positive sample pairs.
        1.0 means positive and otherwise 0.0. It should be symmetric with 1.0 at diagonal. [batch_size, batch_size].

    Returns:
      A scalar for the computed loss.
    """
    batch_size, num_locals, feature_dim = local_features.shape
    num_globals = global_features.shape[-2]

    # Make the input tensors the right shape
    local_features = local_features.view(-1, feature_dim)
    global_features = global_features.view(-1, feature_dim)

    # Compute the outer product, we want a [batch_size, num_locals, batch_size, num_globals] tensor
    product = torch.matmul(local_features, global_features.t())
    product = product.view(batch_size, num_locals, batch_size, num_globals)

    if positive_indicator_matrix is None:
        positive_indicator_matrix = torch.eye(batch_size).to(local_features.device)
    negative_indicator_matrix = 1. - positive_indicator_matrix

    # Compute the positive and negative scores, and average the spatial location
    positive_expectation = compute_positive_expectation(product, measure, reduce_mean=False)
    negative_expectation = compute_negative_expectation(product, measure, reduce_mean=False)

    positive_expectation = torch.mean(positive_expectation, dim=[1, 3])
    negative_expectation = torch.mean(negative_expectation, dim=[1, 3])

    # Mask positive and negative terms
    positive_expectation = torch.sum(positive_expectation * positive_indicator_matrix) / torch.max(torch.sum(positive_indicator_matrix), torch.tensor(1e-12))
    negative_expectation = torch.sum(negative_expectation * negative_indicator_matrix) / torch.max(torch.sum(negative_indicator_matrix), torch.tensor(1e-12))

    return negative_expectation - positive_expectation

### Model for InfoDisentangle ###
class InfoDisentangle(nn.Module):
    """
    Model for InfoDisentangle
    """
    def __init__(self,
                 env,
                 exps,
                 state_dim,
                 obs_dim,
                 act_dim,
                 train_config):
        """Initializer.

        Args:
        env: environment
        exps: buffer data
        state_dim: dimesion of state
        obs_dim: dimension of obs
        act_dim: dimension of action
        train_config: representation config to train model
        """
        super().__init__()
        self.env = env
        self.exps = exps
        self.batch_num = 0
        self.batch_size = train_config.batch_size
        self.num_frames = self.exps.next_mask.shape[0]

        self.dynamics_loss_s_weight = train_config.dynamics_loss_s_weight
        self.dynamics_loss_o_weight = train_config.dynamics_loss_o_weight
        self.reward_loss_weight = train_config.reward_loss_weight
        self.representation_loss_s_weight = train_config.representation_loss_s_weight
        self.representation_loss_o_weight = train_config.representation_loss_o_weight
        self.disentangle_loss_weight = train_config.disentangle_loss_weight

        self.state_embedding_dim = train_config.state_embedding_dim
        self.obs_embedding_dim = train_config.obs_embedding_dim
        assert self.state_embedding_dim == self.obs_embedding_dim # Because using MOE fusion
        subencoder_embedding_dim = self.obs_embedding_dim
        hidden_dim = train_config.hidden_dim
        self.use_cnn = train_config.use_cnn
        if self.use_cnn:
            self.encoder_state = SimpleModelCNN(img_shape=self.env.img_size,
                                                output_dim=self.state_embedding_dim,
                                                channels=train_config.channels,
                                                kernel_size=train_config.kernel_sizes,
                                                strides=train_config.strides,
                                                pool=train_config.pool,
                                                pool_size=train_config.pool_size)
            if self.representation_loss_s_weight != 0:
                self.subencoder_state = SimpleModelCNN(img_shape=self.env.img_size,
                                                   output_dim=self.state_embedding_dim,
                                                   channels=train_config.channels,
                                                   kernel_size=train_config.kernel_sizes,
                                                   strides=train_config.strides,
                                                   pool=train_config.pool,
                                                   pool_size=train_config.pool_size)
                new_module_list = nn.ModuleList()
                for name, block in self.subencoder_state.blocks.named_children():
                    if name not in ["conv0"]:
                        new_module_list.add_module(name, block)
                self.subencoder_state.blocks = new_module_list
            
            self.encoder_obs = SimpleModelCNN(img_shape=self.env.img_size,
                                              output_dim=self.obs_embedding_dim,
                                              channels=train_config.channels,
                                              kernel_size=train_config.kernel_sizes,
                                              strides=train_config.strides,
                                              pool=train_config.pool,
                                              pool_size=train_config.pool_size)
            self.subencoder_obs = SimpleModelCNN(img_shape=self.env.img_size,
                                                 output_dim=self.obs_embedding_dim,
                                                 channels=train_config.channels,
                                                 kernel_size=train_config.kernel_sizes,
                                                 strides=train_config.strides,
                                                 pool=train_config.pool,
                                                 pool_size=train_config.pool_size)
            new_module_list = nn.ModuleList()
            for name, block in self.subencoder_obs.blocks.named_children():
                if name not in ["conv0"]:
                    new_module_list.add_module(name, block)
            self.subencoder_obs.blocks = new_module_list
        else:
            self.encoder_state = SimpleModel(state_dim, self.state_embedding_dim,
                                       hidden_dim=hidden_dim, num_residual_linear_block=1, num_layers_per_block=1,
                                       dropout_rate=train_config.dropout_rate, use_batch_norm=train_config.use_batch_norm)

            self.encoder_obs = SimpleModel(obs_dim, self.obs_embedding_dim,
                                        hidden_dim=hidden_dim, num_residual_linear_block=1, num_layers_per_block=1,
                                        dropout_rate=train_config.dropout_rate, use_batch_norm=train_config.use_batch_norm)
            self.subencoder_obs = SimpleModel(hidden_dim, subencoder_embedding_dim,
                                           hidden_dim=hidden_dim, num_residual_linear_block=1, num_layers_per_block=1,
                                           dropout_rate=train_config.dropout_rate, use_batch_norm=train_config.use_batch_norm)
            new_module_list = nn.ModuleList()
            for name, block in self.subencoder_obs.blocks.named_children():
                if name not in ["fc0"]:
                    new_module_list.add_module(name, block)
            self.subencoder_obs.blocks = new_module_list

        # self.inter_likelihood_estimator_global = CLUB(self.state_embedding_dim, self.obs_embedding_dim, self.obs_embedding_dim)
        self.inter_likelihood_estimator_global = CLUBSample(self.state_embedding_dim, self.obs_embedding_dim, self.obs_embedding_dim)

        self.act_embedding_dim = train_config.act_embedding_dim
        try:
            self.act_embedding = nn.Linear(self.env.action_space.shape[0], self.act_embedding_dim)
        except:
            self.act_embedding = nn.Linear(1, self.act_embedding_dim)
        self.dynamics_model = nn.Sequential(nn.Linear(self.obs_embedding_dim + self.act_embedding_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU())
        self.next_state_model = nn.Linear(hidden_dim, self.state_embedding_dim)
        self.next_obs_model = nn.Linear(hidden_dim, self.obs_embedding_dim)
        self.reward_model = nn.Linear(hidden_dim, 1)
        self.get_optimizer(train_config.lr_encoder, train_config.lr_estimator)

    def get_optimizer(self, learning_rate_encoder, learning_rate_estimator):
        optim_list =  list(self.encoder_state.parameters()) + list(self.subencoder_state.parameters()) if self.representation_loss_s_weight != 0 else list(self.encoder_state.parameters())
        self.encoder_optimizer = torch.optim.Adam(
            optim_list +
            list(self.encoder_obs.parameters()) +
            list(self.subencoder_obs.parameters()) +
            list(self.act_embedding.parameters()) +
            list(self.dynamics_model.parameters()) +
            list(self.next_state_model.parameters()) +
            list(self.next_obs_model.parameters()) +
            list(self.reward_model.parameters()),
            lr=learning_rate_encoder,
            eps=1e-8)

        self.estimator_optimizer = torch.optim.Adam(
            self.inter_likelihood_estimator_global.parameters(),
            lr=learning_rate_estimator,
            eps=1e-8)

    def preprocess(self, x, size):
        """ Transform the input vector to image type

        Args:
            x: input vector (batch, dim)
        Returns:
            x: image (batch, n_channels, size, size)
        """
        return x.reshape(x.shape[0], size[0], size[1], size[2])

    def forward_state(self, states):
        """Computes a forward pass.

        Args:
          states: An input tensor.

        Returns:
          encoder_outputs: An output tensor of the encoder. [batch_size, 1, embedding_dim].
          subencoder_outputs: An output tensor of the subencoder. [batch_size, 1, embedding_dim].
        """
        if self.use_cnn:
            states = self.preprocess(states, self.env.img_size)
            encoder_outputs, encoder_activations = self.encoder_state(states)
            if self.representation_loss_s_weight != 0:
                subencoder_outputs, _ = self.subencoder_state(encoder_activations['conv0'])
        else:
            encoder_outputs, encoder_activations = self.encoder_state(states)
            if self.representation_loss_s_weight != 0:
                subencoder_outputs, _ = self.subencoder_state(encoder_activations['fc0'])
        encoder_outputs = encoder_outputs.unsqueeze(1)
        if self.representation_loss_s_weight != 0:
            subencoder_outputs = subencoder_outputs.unsqueeze(1)
            return encoder_outputs, subencoder_outputs
        else:
            return encoder_outputs, None

    def forward_obs(self, obs):
        """Computes a forward pass.

        Args:
          obs: An input tensor.

        Returns:
          encoder_outputs: An output tensor of the encoder. [batch_size, 1, embedding_dim].
          subencoder_outputs: An output tensor of the subencoder. [batch_size, 1, embedding_dim].
        """
        if self.use_cnn:
            obs = self.preprocess(obs, self.env.img_size)
            encoder_outputs, encoder_activations = self.encoder_obs(obs)
            subencoder_outputs, _ = self.subencoder_obs(encoder_activations['conv0'])
        else:
            encoder_outputs, encoder_activations = self.encoder_obs(obs)
            subencoder_outputs, _ = self.subencoder_obs(encoder_activations['fc0'])
        encoder_outputs = encoder_outputs.unsqueeze(1)
        subencoder_outputs = subencoder_outputs.unsqueeze(1)
        return encoder_outputs, subencoder_outputs
        
    def gen_mask(self, X, plot_name=None):
        """
        X: tensor shape [batch_size, dim1, dim2, ...]
        return: tensor shape [batch_size, batch_size]
        """
        N = X.shape[0]
        reshaped_tensor = X.view(N, -1)
        Positive_N = torch.eq(reshaped_tensor.unsqueeze(1), reshaped_tensor.unsqueeze(0)).all(dim=2).float()
        if plot_name != None:
            plt.imshow(Positive_N.detach().cpu().numpy(), cmap='binary', interpolation='nearest')
            plt.title('Positive Matrix')
            plt.xlabel('Indices')
            plt.ylabel('Indices')
            plt.colorbar()
            plt.savefig(f'{plot_name}.png')
            plt.close()
        return Positive_N

    def update(self):
        """Trains the model for one step. 
        Args:
            e: idx of epoch
        Returns:
            A dictionary for all losses.
        """
        log_encoder_loss = []
        log_encoder_state_representation_loss = []
        log_encoder_obs_representation_loss = []
        log_encoder_disentangle_loss_inter = []
        log_encoder_state_dynamics_loss = []
        log_encoder_obs_dynamics_loss = []
        log_encoder_reward_loss = []

        log_estimator_loss = []

        batch_data = self._get_batches_idxes()
        for inds in tqdm(batch_data):
            encoder_total_loss = 0.0

            sb = self.exps[inds]
            obs_indicator_matrix = self.gen_mask(sb.obs)
            state_indicator_matrix = self.gen_mask(sb.state)

            self.encoder_optimizer.zero_grad()
            # Compute state embeddings, subencoder state embeddings, obs embeddings and subencoder obs embeddings
            state_embeddings, subencoder_state_embeddings = self.forward_state(sb.state)
            obs_embeddings, subencoder_obs_embeddings = self.forward_obs(sb.obs)

            # Compute MI loss between (s; zo+zs) and (o, zo)
            if self.representation_loss_s_weight != 0:
                state_representation_loss = compute_fenchel_dual_loss(subencoder_state_embeddings, state_embeddings, 'JSD', state_indicator_matrix.to(sb.state.device))
                encoder_total_loss += self.representation_loss_s_weight * state_representation_loss
            obs_representation_loss = compute_fenchel_dual_loss(subencoder_obs_embeddings, obs_embeddings, 'JSD', obs_indicator_matrix.to(sb.obs.device))
            encoder_total_loss += self.representation_loss_o_weight * obs_representation_loss
        
            # reshape to (batch_size, embedding_dim)
            state_embeddings = state_embeddings.squeeze(1)
            obs_embeddings = obs_embeddings.squeeze(1)
            fusion_embeddings = 0.5 * (state_embeddings + obs_embeddings)
            fusion_embeddings = fusion_embeddings.squeeze(1)
        
            # Compute dynamic loss
            if len(sb.action.shape) == 1:
                action = sb.action.unsqueeze(1).float()
            else:
                action = sb.action.float()
            act_embeddings = self.act_embedding(action)
            concat_embeddings = torch.cat([fusion_embeddings, act_embeddings], dim=-1)
            dynamics_embeddings = self.dynamics_model(concat_embeddings)
            next_state_preds = self.next_state_model(dynamics_embeddings)
            next_obs_preds = self.next_obs_model(dynamics_embeddings)
            reward_preds = self.reward_model(dynamics_embeddings)

            next_state_targets = self.forward_state(sb.next_state)[0]
            next_state_targets = next_state_targets.squeeze(1)

            next_obs_targets = self.forward_obs(sb.next_obs)[0]
            next_obs_targets = next_obs_targets.squeeze(1)
        
            state_dynamics_loss = torch.pow(torch.norm(next_state_preds - (next_state_targets.detach() * sb.next_mask.unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            obs_dynamics_loss = torch.pow(torch.norm(next_obs_preds - (next_obs_targets.detach() * sb.next_mask.unsqueeze(dim=1)), p=2, dim=1), 2).mean()
            reward_loss = torch.pow(sb.reward.unsqueeze(dim=1) - reward_preds, 2).mean()

            encoder_total_loss += self.dynamics_loss_s_weight * state_dynamics_loss
            encoder_total_loss += self.dynamics_loss_o_weight * obs_dynamics_loss
            encoder_total_loss += self.reward_loss_weight * reward_loss

            # Compute disentagle loss MI(zs; zo)
            self.inter_likelihood_estimator_global.eval()
            inter_bound_global = self.inter_likelihood_estimator_global.mi_est(state_embeddings, obs_embeddings)
            encoder_total_loss += self.disentangle_loss_weight * inter_bound_global

            # Compute encoder gradients
            encoder_total_loss.backward()
            self.encoder_optimizer.step()
          
            log_encoder_loss.append(encoder_total_loss.item())
            if self.representation_loss_s_weight != 0:
                log_encoder_state_representation_loss.append(state_representation_loss.item())
            log_encoder_obs_representation_loss.append(obs_representation_loss.item())
            log_encoder_disentangle_loss_inter.append(inter_bound_global.item())
            log_encoder_state_dynamics_loss.append(state_dynamics_loss.item())
            log_encoder_obs_dynamics_loss.append(obs_dynamics_loss.item())
            log_encoder_reward_loss.append(reward_loss.item())

            estimator_total_loss = 0.
            for _ in range(5):
                self.inter_likelihood_estimator_global.train()
                rand_id = np.random.randint(0, self.num_frames // self.batch_size)
                sb_mi = self.exps[batch_data[rand_id]]
                state_embeddings = self.forward_state(sb_mi.state)[0]
                state_embeddings = state_embeddings.squeeze(1)
                obs_embeddings = self.forward_obs(sb_mi.obs)[0]
                obs_embeddings = obs_embeddings.squeeze(1)
                likelihood_loss = -self.inter_likelihood_estimator_global.loglikeli(state_embeddings, obs_embeddings)
                self.estimator_optimizer.zero_grad()
                likelihood_loss.backward()
                self.estimator_optimizer.step()
                estimator_total_loss += likelihood_loss.item()
            log_estimator_loss.append(estimator_total_loss / 5)

        encoder_losses = dict(
            total_loss=np.mean(log_encoder_loss),
            max_s_zs=np.mean(log_encoder_state_representation_loss),
            max_o_zo=np.mean(log_encoder_obs_representation_loss),
            min_zs_zo=np.mean(log_encoder_disentangle_loss_inter),
            state_dynamics_loss=np.mean(log_encoder_state_dynamics_loss),
            obs_dynamics_loss=np.mean(log_encoder_obs_dynamics_loss),
            reward_loss=np.mean(log_encoder_reward_loss),
        )

        estimator_losses = dict(
            mi_zs_zo=np.mean(log_estimator_loss),
        )

        return dict(
            encoder=encoder_losses,
            estimator=estimator_losses,
        )

    def _get_batches_idxes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = np.arange(0, self.num_frames)
        indexes = np.random.permutation(indexes)
        self.batch_num += 1

        num_indexes = self.batch_size
        batches_starting_indexes = [indexes[i:i+num_indexes]
                                    for i in range(0, len(indexes), num_indexes)]
        # drop the last one if it is not full
        if len(batches_starting_indexes[-1]) < self.batch_size:
            batches_starting_indexes = batches_starting_indexes[:-1]
        return batches_starting_indexes

    def save(self, path, to_cloud=False):
        torch.save(self.encoder_state.state_dict(),
                   os.path.join(path, "s_encoder.pt"))
        torch.save(self.encoder_obs.state_dict(),
                   os.path.join(path, "o_encoder.pt"))

        if to_cloud:
            wandb.save(os.path.join(path, "s_encoder.pt"))
            wandb.save(os.path.join(path, "o_encoder.pt"))