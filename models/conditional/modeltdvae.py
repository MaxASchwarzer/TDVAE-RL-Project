import collections
import glob
import pathlib

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.nn import functional as F
import numpy as np

from pylego import ops, misc

from ..baseconditional import BaseGymTDVAE
from .utils import Discriminator, SAGANGenerator
from .gymtdvae import ConvPreProcess, PreProcess, DBlock


class OptionProcessor(nn.Module):
    def __init__(self, z_dim, h_dim, indim):
        super().__init__()
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.input_fc = nn.Linear(indim, h_dim)

        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)

    def forward(self, parameters):
        current = self.input_fc(parameters)
        current = self.bn1(self.relu(current.view(-1, current.shape[-1]))).view(*current.shape)
        current = self.fc1(current)
        current = self.bn2(self.relu(current.view(-1, current.shape[-1]))).view(*current.shape)
        current = self.fc2(current)
        return current


class HyperOption(nn.Module):
    def __init__(self, z_dim, h_dim, a_dim, inner_h_dim=16, layers=1, distributional=False):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.output_fcs = nn.ModuleList()
        if layers == 1:
            self.total_sizes = [z_dim*a_dim, a_dim]
        else:
            self.total_sizes = [z_dim * inner_h_dim, inner_h_dim,]
            for i in range(layers - 2):
                self.total_sizes += [inner_h_dim**2, inner_h_dim]
            self.total_sizes += [inner_h_dim*a_dim, a_dim]

        self.output_fc = nn.Linear(h_dim, int(np.sum(self.total_sizes)))
        self.output_var_fc = nn.Linear(h_dim, int(np.sum(self.total_sizes)))

        self.distributional = distributional

        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)

    def forward(self, state):
        current = self.fc1(state)
        current = self.bn1(self.relu(current).view(-1, current.shape[-1])).view(*current.shape)
        current = self.fc2(current)
        current = self.bn2(self.relu(current).view(-1, current.shape[-1])).view(*current.shape)

        means = self.output_fc(current)

        variances = self.output_var_fc(current) if self.distributional else None

        return means, variances, self.total_sizes


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def apply_option(state, option, sizes):
    current = state
    option_params = torch.split(option, sizes, -1)
    for i, (weight, bias) in enumerate(chunks(option_params, 2)):
        weight = weight.view(option.shape[0], -1, current.shape[-1])
        current = current.unsqueeze(-1)
        while len(weight.shape) < len(current.shape):
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        current = torch.matmul(weight, current).squeeze(-1) + bias
        if i < len(option_params)//2 - 1:
            current = F.relu(current)

    return current


class OptionInferenceNetwork(nn.Module):
    def __init__(self, zdim, adim, hdim, inner_h_dim, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(adim, hdim//4)
        self.recurrent = nn.LSTM(zdim+hdim//4, hdim, bidirectional=True, num_layers=num_layers)
        self.final = nn.Linear(hdim*num_layers*2, zdim)
        self.hyperoption = HyperOption(zdim, hdim, adim, inner_h_dim, layers=num_layers, distributional=True)

    def forward(self, states, a, tdiffs):
        a = self.emb(a)
        inputs = torch.cat([states, a], -1).transpose(0, 1)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, tdiffs)
        outputs = self.recurrent(packed)[1][0]
        outputs = outputs.transpose(0, 1).flatten(1, -1)
        processed = self.final(outputs)

        mean, variance, sizes = self.hyperoption(processed)

        return mean, variance, sizes


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_size, inner_hidden, discrete_actions=False, base=None, base_kwargs=None):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if discrete_actions:
            self.hyper = False
            self.dist_type = Categorical
            self.base = AdvantageNetwork(obs_shape, hidden_size, action_dim)
            self.value_base = ValueNetwork(obs_shape, hidden_size)
        else:
            self.hyper = True
            self.base = HyperOption(obs_shape, hidden_size, action_dim, inner_hidden, **base_kwargs)
            self.value_base = ValueNetwork(obs_shape, hidden_size)
            self.dist_type = lambda mean, sigma: Independent(Normal(mean, sigma), 1)

    def act(self, inputs, deterministic=False, option=None):
        if option is None:
            mean, logvar, sizes = self.base(inputs)
            dist = self.dist(mean, logvar.exp())
            if deterministic:
                option = dist.mean()
            else:
                option = dist.sample()
            option_log_probs = dist.log_prob(option)

        else:
            sizes = self.base.total_sizes
            option_log_probs = None

        value = self.value_base(inputs)
        if self.hyper:
            action = apply_option(inputs, option, sizes)

        return value, action, option_log_probs

    def get_value(self, inputs):
        value = self.value_base(inputs)
        return value

    def evaluate_options(self, inputs, options):
        mean, logvar, sizes = self.base(inputs)
        dist = self.dist(mean, logvar.exp())
        value = self.value_base(inputs)

        action_log_probs = dist.log_prob(options)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class ReturnsDecoder(nn.Module):

    def __init__(self, z_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        p = self.fc3(t)
        return p


class AdvantageNetwork(nn.Module):

    def __init__(self, z_size, hidden_size, action_space):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        p = self.fc3(t)
        return p


class ValueNetwork(nn.Module):

    def __init__(self, z_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, z):
        t = F.elu(self.fc1(z))
        t = F.elu(self.fc2(t))
        p = self.fc3(t)
        return p


class QNetwork(nn.Module):

    def __init__(self, z_size, hidden_size, action_space):
        super().__init__()
        self.adv = AdvantageNetwork(z_size, hidden_size, action_space)
        self.val = ValueNetwork(z_size, hidden_size)

    def forward(self, z):
        adv = self.adv(z)  # size: bs, action_space
        adv = adv - adv.mean(dim=1, keepdim=True)
        val = self.val(z)  # size: bs, 1
        return adv + val


class TDQVAE(nn.Module):
    """ The full TD-VAE model with jumpy prediction.
    """

    def __init__(self, x_size, resnet_hidden_size, processed_x_size, b_size, z_size, layers, samples_per_seq,
                 t_diff_min, t_diff_max, t_diff_max_poss=10, action_space=0, action_dim=8, rl=False, model_based=True):
        super().__init__()
        self.layers = layers
        self.samples_per_seq = samples_per_seq
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max
        self.t_diff_max_poss = t_diff_max_poss

        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        self.rl = rl
        self.model_based = model_based

        # Input pre-process layer
        if len(x_size) > 1:
            self.process_x = ConvPreProcess(x_size, resnet_hidden_size, processed_x_size)
        else:
            self.process_x = PreProcess(x_size, processed_x_size)

        # Multilayer LSTM for aggregating belief states
        self.b_rnn = ops.MultilayerLSTM(input_size=processed_x_size+action_dim+1, hidden_size=b_size, layers=layers,
                                        every_layer_input=True, use_previous_higher=True)

        # Multilayer state model is used. Sampling is done by sampling higher layers first.
        self.z_b = nn.ModuleList([DBlock(b_size + (z_size if layer < layers - 1 else 0), layers * b_size, z_size)
                                  for layer in range(layers)])

        # Given belief and state at time t2, infer the state at time t1
        self.z1_z2_b = nn.ModuleList([DBlock(b_size + layers * z_size + (z_size if layer < layers - 1 else 0)
                                             + (t_diff_max_poss - t_diff_min), layers * b_size, z_size)
                                      for layer in range(layers)])

        # Given the state at time t1, model state at time t2 through state transition
        self.z2_z1 = nn.ModuleList([DBlock(layers * z_size + z_size + #extra z_size is option
                                           (z_size if layer < layers - 1 else 0) + (t_diff_max_poss - t_diff_min),
                                           layers * b_size, z_size)
                                    for layer in range(layers)])

        # state to observation
        # self.x_z = ConvDecoder(layers * z_size, resnet_hidden_size, x_size)
        self.x_z = SAGANGenerator(x_size, z_dim=layers * z_size, d_hidden=resnet_hidden_size)

        # state to Q value per action
        if rl:
            self.g_z = ReturnsDecoder((layers * z_size * 2) + z_size + (t_diff_max_poss - t_diff_min),
                                      layers * b_size)
            if not model_based:
                self.q_z = QNetwork(layers * z_size, layers * b_size, action_space)
            else:
                self.actor_critic = ActorCritic(layers*z_size, action_space, b_size, z_size)

        self.option_sizes = self.actor_critic.base.total_sizes
        total_size = int(np.sum(self.actor_critic.base.total_sizes))
        self.action_embedding = nn.Embedding(action_space, action_dim)
        self.action_reconstruction = OptionInferenceNetwork(z_size*layers, action_space, b_size, z_size)
        self.option_embedding = OptionProcessor(z_size, b_size, total_size)

        self.time_encoding = nn.Embedding(t_diff_max_poss - t_diff_min + 1, t_diff_max_poss - t_diff_min)
        for param in self.time_encoding.parameters():
            param.requires_grad = False
            param.zero_()
        for i in range(t_diff_max_poss - t_diff_min + 1):
            self.time_encoding.weight[i, :i] = 1.0
        self.time_encoding_scale = nn.Parameter(torch.ones(1))

    def compute_q(self, x, actions, rewards, done):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])

        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        if actions is not None:
            rewards = (rewards[..., None] / 10.0).clamp(0.0, 2.0)
            action_embs = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, action_embs, rewards], -1)

        # aggregate the belief b
        b = self.b_rnn(processed_x, done)  # size: bs, time, layers, dim
        b = b[:, -1]  # size: bs, layers, dim

        zs = []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                z_mu, z_logvar = self.z_b[layer](b[:, layer])
            else:
                z_mu, z_logvar = self.z_b[layer](torch.cat([b[:, layer], z], dim=1))

            z = ops.reparameterize_gaussian(z_mu, z_logvar, self.training)
            zs.insert(0, z)

        z = torch.cat(zs, dim=1)
        return self.actor_critic.get_value(z)

    def option_reconstruction(self, b, actions, t1, t2):

        b = b.detach()
        qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](b[:, :, layer])
            else:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](torch.cat([b[:, :, layer], qb_z2_b2], dim=-1))
            qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
            qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

            qb_z2_b2 = ops.reparameterize_gaussian(qb_z2_b2_mu, qb_z2_b2_logvar, self.training)
            qb_z2_b2s.insert(0, qb_z2_b2)

        zs = torch.cat(qb_z2_b2s, -1)

        lengths = (t2-t1).squeeze(0)
        sorted_lengths, argsort = lengths.sort(descending=True)
        maxlen = torch.max(lengths, 0)[0].item()
        indices = t1[0, :, None] + torch.arange(0, maxlen, device=b.device)[None, :].expand(zs.shape[0], -1)
        indices = indices[:, :, None]
        indices = indices.clamp(0, b.shape[1]-1)
        b_indices = indices.expand(-1, -1, zs.shape[-1])
        states = torch.gather(zs, 1, b_indices)
        actions = torch.gather(actions, 1, indices.squeeze(-1))
        sorted_states = states[argsort]
        sorted_actions = actions[argsort]

        mean, logvar, sizes = self.action_reconstruction(sorted_states, sorted_actions, sorted_lengths)
        mean = mean.contiguous()
        logvar = logvar.contiguous()
        parameters = ops.reparameterize_gaussian(mean, logvar, self.training)
        inferred_actions = apply_option(states, parameters, sizes)

        reconstruction_loss = F.cross_entropy(inferred_actions.transpose(-1, -2), actions, reduction="none")
        reconstruction_loss = torch.gather(torch.cumsum(reconstruction_loss, 1), 1, (lengths-1)[:, None])

        _, unsort = argsort.sort()
        parameters = parameters[unsort]

        return parameters.detach(), reconstruction_loss, mean, logvar

    def q_and_z_b(self, x, actions, rewards, done, t1, t2):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        if actions is not None:
            rewards = (rewards[..., None] / 10.0).clamp(0.0, 2.0)
            action_embs = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, action_embs, rewards], -1)
        else:
            action_embs = None

        # aggregate the belief b
        b = self.b_rnn(processed_x, done)  # size: bs, time, layers, dim

        # replicate b multiple times
        b_exp = b[None, ...].expand(self.samples_per_seq, -1, -1, -1, -1)  # size: copy, bs, time, layers, dim_exp

        # Element-wise indexing. sizes: bs, layers, dim
        b1 = torch.gather(b_exp, 2, t1[..., None, None, None].expand(-1, -1, -1, b_exp.size(3), b_exp.size(4))).view(
            -1, b_exp.size(3), b_exp.size(4))
        b2 = torch.gather(b_exp, 2, t2[..., None, None, None].expand(-1, -1, -1, b_exp.size(3), b_exp.size(4))).view(
            -1, b_exp.size(3), b_exp.size(4))

        # q_B(z2 | b2)
        qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](b2[:, layer])
            else:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](torch.cat([b2[:, layer], qb_z2_b2], dim=1))
            qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
            qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

            qb_z2_b2 = ops.reparameterize_gaussian(qb_z2_b2_mu, qb_z2_b2_logvar, self.training)
            qb_z2_b2s.insert(0, qb_z2_b2)

        qb_z2_b2_mu = torch.cat(qb_z2_b2_mus, dim=1)
        qb_z2_b2_logvar = torch.cat(qb_z2_b2_logvars, dim=1)
        qb_z2_b2 = torch.cat(qb_z2_b2s, dim=1)

        # p_B(z1 | b1)
        pb_z1_b1_mus, pb_z1_b1_logvars, pb_z1_b1s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](b1[:, layer])
            else:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](torch.cat([b1[:, layer], pb_z1_b1s[0]], dim=1))
            pb_z1_b1_mus.insert(0, pb_z1_b1_mu)
            pb_z1_b1_logvars.insert(0, pb_z1_b1_logvar)
            pb_z1_b1s.insert(0, ops.reparameterize_gaussian(pb_z1_b1_mu, pb_z1_b1_logvar, self.training))

        pb_z1_b1_mu = torch.cat(pb_z1_b1_mus, dim=1)
        pb_z1_b1_logvar = torch.cat(pb_z1_b1_logvars, dim=1)
        pb_z1_b1 = torch.cat(pb_z1_b1s, dim=1)

        if self.rl:
            # Q values
            if not self.model_based:
                q1 = self.q_z(pb_z1_b1_mu)
                q2 = self.q_z(qb_z2_b2_mu)
            else:
                q1 = self.actor_critic.get_value(pb_z1_b1_mu)
                q2 = self.actor_critic.get_value(qb_z2_b2_mu)

        else:
            q1, q2 = 0, 0

        return q1, q2, action_embs, b1, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2s, qb_z2_b2, pb_z1_b1_mu, pb_z1_b1_logvar,\
               pb_z1_b1, b

    def predict_forward(self, qs_z1_b1, option, t_encodings):
        pt_z2_z1_mus, pt_z2_z1_logvars, pt_z2_z1s = [], [], []
        option = self.option_embedding(option)
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_b1, t_encodings, option], dim=-1))
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_b1, pt_z2_z1s[0],
                                                                            t_encodings, option], dim=-1))
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)
            pt_z2_z1s.insert(0, ops.reparameterize_gaussian(pt_z2_z1_mu, pt_z2_z1_logvar, self.training))

        pt_z2_z1 = torch.cat(pt_z2_z1s, dim=-1)
        pd_g2_z2_mu = self.g_z(torch.cat([qs_z1_b1, pt_z2_z1, option, t_encodings], dim=-1))
        value = self.actor_critic.get_value(pt_z2_z1)

        return pt_z2_z1, pd_g2_z2_mu, value

    def forward(self, x, actions, rewards, done, t1, t2):
        if t1 is None:
            t1 = torch.randint(0, x.size(1) - int(self.rl) - self.t_diff_max, (self.samples_per_seq, x.size(0)),
                               device=x.device)
        else:
            t1 = t1[None, :]
        if t2 is None:
            t2 = t1 + torch.randint(self.t_diff_min, self.t_diff_max + 1, (self.samples_per_seq, x.size(0)),
                                    device=x.device)
        else:
            t2 = t2[None, :]

        q1, q2, action_embs, b1, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2s, qb_z2_b2, pb_z1_b1_mu, pb_z1_b1_logvar, \
            pb_z1_b1, b = self.q_and_z_b(x, actions, rewards, done, t1, t2)

        option, option_reconstruction_loss, o_mean, o_logvar = self.option_reconstruction(b, actions, t1, t2)

        t_encodings = self.time_encoding(t2 - t1 - self.t_diff_min) * self.time_encoding_scale
        t_encodings = t_encodings.view(-1, t_encodings.size(-1))
        if action_embs is not None:
            t1_next = t1 + 1
            action_embs = action_embs[None, ...].expand(self.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
            a1_next = torch.gather(action_embs, 2, t1_next[..., None, None].expand(
                -1, -1, -1, action_embs.shape[-1])).view(-1, action_embs.shape[-1])

        # q_S(z1 | z2, b1, b2) ~= q_S(z1 | z2, b1)
        qs_z1_z2_b1_mus, qs_z1_z2_b1_logvars, qs_z1_z2_b1s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b[layer](torch.cat([qb_z2_b2, b1[:, layer],
                                                                                    t_encodings], dim=1))
            else:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b[layer](torch.cat([qb_z2_b2, b1[:, layer],
                                                                                    qs_z1_z2_b1, t_encodings],
                                                                                   dim=1))
            qs_z1_z2_b1_mus.insert(0, qs_z1_z2_b1_mu)
            qs_z1_z2_b1_logvars.insert(0, qs_z1_z2_b1_logvar)

            qs_z1_z2_b1 = ops.reparameterize_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, self.training)
            qs_z1_z2_b1s.insert(0, qs_z1_z2_b1)

        qs_z1_z2_b1_mu = torch.cat(qs_z1_z2_b1_mus, dim=1)
        qs_z1_z2_b1_logvar = torch.cat(qs_z1_z2_b1_logvars, dim=1)
        qs_z1_z2_b1 = torch.cat(qs_z1_z2_b1s, dim=1)

        # p_T(z2 | z1), also conditions on q_B(z2) from higher layer
        emb_option = self.option_embedding(option)
        pt_z2_z1_mus, pt_z2_z1_logvars = [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_z2_b1, t_encodings, emb_option], dim=1))
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_z2_b1, qb_z2_b2s[layer + 1],
                                                                            t_encodings, emb_option], dim=1))
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)

        pt_z2_z1_mu = torch.cat(pt_z2_z1_mus, dim=1)
        pt_z2_z1_logvar = torch.cat(pt_z2_z1_logvars, dim=1)

        # p_D(x2 | z2)
        pd_x2_z2 = self.x_z(qb_z2_b2)
        # p_D(g2 | z1, z2, a1', t2-t1)
        if self.rl:
            pd_g2_z2_mu = self.g_z(torch.cat([qs_z1_z2_b1, qb_z2_b2, emb_option, t_encodings], dim=1))
        else:
            pd_g2_z2_mu = None

        return (x, actions, option, rewards, done, t1, t2, t_encodings, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu,
                pb_z1_b1_logvar, pb_z1_b1, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar,
                pd_x2_z2, pd_g2_z2_mu, q1, q2, option_reconstruction_loss, o_mean, o_logvar, option)

    def visualize(self, x, t, n, actions, rewards, done):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)

        if actions is not None:
            rewards = (rewards[..., None] / 10.0).clamp(0.0, 2.0)
            action_embs = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, action_embs, rewards], -1)
        else:
            action_embs = None

        # aggregate the belief b
        full_b = self.b_rnn(processed_x, done)# size: bs, time, layers, dim
        b = full_b[:, t]  # Just pick out relevant time
        t_encodings = self.time_encoding(b.new_zeros(b.size(0), dtype=torch.long)) * self.time_encoding_scale
        t1 = torch.zeros(x.shape[0], device=x.device)
        t2 = torch.zeros(x.shape[0], device=x.device) + x.shape[1] - 1
        option, _, _, _ = self.option_reconstruction(full_b, actions, t1.unsqueeze(0).long(), t2.unsqueeze(0).long())
        processed_option = self.option_embedding(option)
        # compute z from b
        p_z_bs = []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](b[:, layer])
            else:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](torch.cat([b[:, layer], p_z_b], dim=1))
            p_z_b = ops.reparameterize_gaussian(p_z_b_mu, p_z_b_logvar, True)
            p_z_bs.insert(0, p_z_b_mu)

        z = torch.cat(p_z_bs, dim=1)
        rollout_x = [self.x_z(z)]
        for i in range(n - 1):
            next_z = []
            for layer in range(self.layers - 1, -1, -1):
                if layer == self.layers - 1:
                    if actions is not None:
                        inputs = torch.cat([z, t_encodings, processed_option], dim=1)
                    else:
                        inputs = torch.cat([z, t_encodings], dim=1)
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](inputs)
                else:
                    if actions is not None:
                        inputs = torch.cat([z, pt_z2_z1, t_encodings, processed_option], dim=1)
                    else:
                        inputs = torch.cat([z, pt_z2_z1, t_encodings], dim=1)
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](inputs)
                pt_z2_z1 = ops.reparameterize_gaussian(pt_z2_z1_mu, pt_z2_z1_logvar, True)
                next_z.insert(0, pt_z2_z1_mu)

            z = torch.cat(next_z, dim=1)
            rollout_x.append(self.x_z(z))

        return torch.stack(rollout_x, dim=1)

    def predictive_control(self, x, actions, done, rewards, num_rollouts=100, rollout_length=1,
                           jump_length=10, gamma=0.99, boltzmann=True):
        with torch.no_grad():
            # pre-process image x
            im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
            processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
            processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
            if actions is not None:
                rewards = (rewards[..., None] / 10.0).clamp(0.0, 2.0)
                action_embs = self.action_embedding(actions)
                processed_x = torch.cat([processed_x, action_embs, rewards], -1)
            else:
                action_embs = None

            # aggregate the belief b
            # size: bs, rollout, layers, dim
            b = self.b_rnn(processed_x, done)[:, -1][:, None].expand(-1, num_rollouts, -1, -1)

            # q_B(z2 | b2)
            qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
            for layer in range(self.layers - 1, -1, -1):
                if layer == self.layers - 1:
                    qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](b[:, :, layer])
                else:
                    qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](torch.cat([b[:, :, layer], qb_z2_b2], dim=-1))
                qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
                qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

                qb_z2_b2 = ops.reparameterize_gaussian(qb_z2_b2_mu, qb_z2_b2_logvar, self.training)
                qb_z2_b2s.insert(0, qb_z2_b2)

            initial = torch.cat(qb_z2_b2s, dim=-1)[:, -1].unsqueeze(1).expand(-1, num_rollouts, -1)

            distributions = Normal(0, 1)
            sizes = self.actor_critic.base.total_sizes
            parameters = distributions.sample((x.shape[0], num_rollouts, int(np.sum(sizes)))).to(b.device)

            current = initial
            running = 0
            jump_encoding = torch.tensor([jump_length], device=b.device, dtype=torch.long)
            jump_encoding = jump_encoding[None, None, :].expand(current.shape[0], current.shape[1], -1)
            jump_encoding = self.time_encoding(jump_encoding).squeeze(-2)
            for i in range(rollout_length):
                current, pd_g2, value = self.predict_forward(current, parameters, jump_encoding)

                pd_g2 = pd_g2*(gamma ** (i*jump_length))
                running = pd_g2 + running

            running = value*(gamma ** ((i+1)*jump_length)) + running

            # print(running[0].mean().item(), running[0].var().item(), running[0].max().item(), running[0].min().item())
            best = torch.max(running, 1)[1]
            indices = best[:, None,].expand(-1, -1, parameters.shape[-1])
            best_option = torch.gather(parameters, 1, indices).squeeze(-2)
            action = apply_option(initial[:, 0], best_option, sizes)
            if not boltzmann:
                action = torch.max(action, -1)[1]
            else:
                dist = Categorical(logits = action)
                action = dist.sample()

        return action.cpu().numpy()


class GymTDQVAE(BaseGymTDVAE):

    def __init__(self, flags, model=None, action_space=20, rl=False, replay_buffer=None, *args, **kwargs):
        self.rl = rl
        self.model_based = True
        self.replay_buffer = replay_buffer

        self.adversarial = flags.adversarial
        self.beta_decay = misc.LinearDecay(flags.beta_decay_start, flags.beta_decay_end, flags.beta_initial, flags.beta)

        self.d_weight = flags.d_weight
        self.d_steps = flags.d_steps
        if flags.t_diff_max_poss < 0:
            t_diff_max_poss = flags.seq_len - 1
        else:
            t_diff_max_poss = flags.t_diff_max_poss

        if model is None:
            model_args = [(1, 40, 40), flags.h_size, 2*flags.b_size, flags.b_size, flags.z_size, flags.layers,
                          flags.samples_per_seq, flags.t_diff_min, flags.t_diff_max, t_diff_max_poss]
            model_kwargs = {'action_space': action_space}
            if rl:
                model_kwargs['rl'] = True
            model = TDQVAE(*model_args, **model_kwargs)
        if self.adversarial:
            kwargs['optimizer'] = None  # we create an optimizer later
        super().__init__(model, flags, *args, **kwargs)

        if self.adversarial:
            self.dnet = Discriminator(channels=3, d_size=flags.d_size)
            self.dnet.to(self.device)

            self.adversarial_optim = torch.optim.Adam(self.dnet.parameters(), lr=flags.d_lr, betas=(0.0, 0.9))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=flags.learning_rate, betas=(0.0, 0.9))
            flags.optimizer = self.optimizer

        if rl:
            self.target_net = TDQVAE(*model_args, **model_kwargs)
            self.target_net.eval()
            self.target_net.to(self.device)
        else:
            self.target_net = None

        if flags.load_file:
            self.load(flags.load_file)

    def update_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def loss_function(self, forward_ret, labels=None, loss=F.binary_cross_entropy):
        (x_orig, actions, options, rewards, done, t1, t2, t_encodings, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu,
         pb_z1_b1_logvar, pb_z1_b1, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2,
         pd_g2_z2_mu, q1, q2, option_recon_loss, o_mean, o_logvar, option) = forward_ret

        # replicate x multiple times
        x = x_orig.flatten(2, -1)
        x = x[None, ...].expand(self.flags.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
        x2 = torch.gather(x, 2, t2[..., None, None].expand(-1, -1, -1, x.size(3))).view(-1, x.size(3))
        kl_div_qs_pb = ops.kl_div_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar)
        # kl_div_option = ops.kl_div_gaussian(o_mean, o_logvar)
        kl_div_option = 0.5*torch.sum(o_mean**2 + o_logvar.exp() - o_logvar - 1)

        kl_shift_qb_pt = (ops.gaussian_log_prob(qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2) -
                          ops.gaussian_log_prob(pt_z2_z1_mu, pt_z2_z1_logvar, qb_z2_b2))

        pd_x2_z2 = pd_x2_z2.flatten(1, -1)
        bce = loss(pd_x2_z2, x2, reduction='none').sum(dim=1)
        bce_optimal = loss(x2, x2, reduction='none').sum(dim=1)
        bce_diff = bce - bce_optimal

        if self.adversarial and self.is_training():
            r_in = x2.view(x2.shape[0], x.shape[2], x.shape[3], x.shape[4])
            f_in = pd_x2_z2.view(x2.shape[0], x.shape[2], x.shape[3], x.shape[4])
            for _ in range(self.d_steps):
                d_loss, g_loss, hidden_loss = self.dnet.get_loss(r_in, f_in)
                d_loss.backward(retain_graph=True)

                self.adversarial_optim.step()
                self.adversarial_optim.zero_grad()
            bce_diff = hidden_loss  # XXX bce_diff added twice to loss?
        else:
            g_loss = 0
            hidden_loss = 0

        if self.model_based:
            # pred_z2, pred_g = self.model.predict_forward(pb_z1_b1, options, t_encodings)
            # with torch.no_grad():
            #     # size: bs, action_space
            #     t1_next = t1 + 1
            #     t2_next = t2 + 1
            #     _, pred_values = self.target_net.q_and_z_b(x_orig, actions, rewards, done, t1_next,
            #                                                t2_next)[:2]
            #
            # target_q2 = r2_next + self.flags.discount_factor * (1.0 - done2_next) * q2_next

            # Note: x[t], rewards[t] is a result of actions[t]
            # Q(s[t], a[t+1]) = r[t+1] + γ max_a Q(s[t+1], a)
            returns, is_weight = labels

            # use pd_g2_z2_mu for returns modeling
            returns_loss = (pd_g2_z2_mu.squeeze(1) - (10.0 * returns)) ** 2

            # XXX reward clipping hardcoded for Seaquest
            clipped_rewards = (rewards / 10.0).clamp(0.0, 2.0)

            t1_next = t1 + 1
            t2_next = t2 + 1

            with torch.no_grad():
                # size: bs, action_space
                q1_next_target, q2_next_target = self.target_net.q_and_z_b(x_orig, actions, rewards, done, t1_next,
                                                                           t2_next)[:2]
                q1_next_index, q2_next_index = self.model.q_and_z_b(x_orig, actions, rewards, done, t1_next,
                                                                    t2_next)[:2]
                q1_next_index = torch.argmax(q1_next_index, dim=1, keepdim=True)
                q2_next_index = torch.argmax(q2_next_index, dim=1, keepdim=True)

            done = done[None, ...].expand(self.flags.samples_per_seq, -1, -1)  # size: copy, bs, time
            done1_next = torch.gather(done, 2, t1_next[..., None]).view(-1)  # size: bs
            done2_next = torch.gather(done, 2, t2_next[..., None]).view(-1)  # size: bs

            # size: copy, bs, time
            clipped_rewards = clipped_rewards[None, ...].expand(self.flags.samples_per_seq, -1, -1)
            r1_next = torch.gather(clipped_rewards, 2, t1_next[..., None]).view(-1)  # size: bs
            r2_next = torch.gather(clipped_rewards, 2, t2_next[..., None]).view(-1)  # size: bs

            # actions = actions[None, ...].expand(self.flags.samples_per_seq, -1, -1)  # size: copy, bs, time
            # a1_next = torch.gather(actions, 2, t1_next[..., None]).view(-1)  # size: bs
            # a2_next = torch.gather(actions, 2, t2_next[..., None]).view(-1)  # size: bs
            #
            # pred_q1 = torch.gather(q1, 1, a1_next[..., None]).view(-1)
            # pred_q2 = torch.gather(q2, 1, a2_next[..., None]).view(-1)

            q1 = q1.squeeze(-1)
            q2 = q2.squeeze(-1)
            q1_next = torch.gather(q1_next_target, 1, q1_next_index).view(-1)
            q2_next = torch.gather(q2_next_target, 1, q2_next_index).view(-1)
            target_q1 = r1_next + self.flags.discount_factor * (1.0 - done1_next) * q1_next
            target_q2 = r2_next + self.flags.discount_factor * (1.0 - done2_next) * q2_next
            print(q1[0], target_q1[0])
            rl_loss = 0.5 * (F.smooth_l1_loss(q1, target_q1, reduction='none') +
                             F.smooth_l1_loss(q2, target_q2, reduction='none'))
            # errors for prioritized experience replay
            rl_errors = 0.5 * (torch.abs(q1 - target_q1) + torch.abs(q2 - target_q2)).detach()

        else:
            returns_loss = 0.0
            rl_loss = 0.0
            is_weight = 1.0
            rl_errors = 0.0

        # multiply is_weight separately for ease of reporting
        is_weight = is_weight.float()
        returns_loss = is_weight * returns_loss
        bce_optimal = is_weight * bce_optimal
        bce_diff = is_weight * bce_diff
        hidden_loss = is_weight * hidden_loss
        g_loss = is_weight * g_loss
        kl_div_qs_pb = is_weight * kl_div_qs_pb
        kl_shift_qb_pt = is_weight * kl_shift_qb_pt
        rl_loss = is_weight * rl_loss

        beta = self.beta_decay.get_y(self.get_train_steps())
        tdvae_loss = bce_diff + returns_loss + hidden_loss + self.d_weight * g_loss + beta * (kl_div_qs_pb +
                                                                                              kl_shift_qb_pt)
        option_loss = option_recon_loss + beta*kl_div_option*0.01
        loss = self.flags.tdvae_weight * tdvae_loss + self.flags.rl_weight * rl_loss + option_loss

        if self.rl:  # workaround to work with non-RL setting
            rl_loss = rl_loss.mean()
            returns_loss = returns_loss.mean()
        return collections.OrderedDict([('loss', loss.mean()),
                                        ('bce_diff', bce_diff.mean()),
                                        ('returns_loss', returns_loss),
                                        ('kl_div_qs_pb', kl_div_qs_pb.mean()),
                                        ('kl_shift_qb_pt', kl_shift_qb_pt.mean()),
                                        ('kl_div_option', kl_div_option.mean()),
                                        ('reconstruction_option', option_recon_loss.mean()),
                                        ('rl_loss', rl_loss),
                                        ('bce_optimal', bce_optimal.mean()),
                                        ('rl_errors', rl_errors)])

    def initialize(self, load_file):
        '''Overriding: do not load file during superclass initialization, we do it manually later in init'''
        pass

    def load(self, load_file):
        """Load a model from a saved file."""
        print("* Loading model from", load_file, "...")
        m_state_dict, target_net_state_dict, o_state_dict, train_steps, replay_buffer = torch.load(load_file)
        self.model.load_state_dict(m_state_dict)
        if self.target_net is not None and target_net_state_dict is not None:
            self.target_net.load_state_dict(target_net_state_dict)
            self.replay_buffer.load_buffer(replay_buffer)
        self.optimizer.load_state_dict(o_state_dict)
        self.train_steps = train_steps
        if self.adversarial:
            m_state_dict, o_state_dict = torch.load(load_file+"disc")
            self.dnet.load_state_dict(m_state_dict)
            self.adversarial_optim.load_state_dict(o_state_dict)
        print("* Loaded model from", load_file)

    def save(self, save_file):
        "Save model to file."
        save_fname = save_file + "." + str(self.train_steps)
        print("* Saving model to", save_fname, "...")
        existing = glob.glob(save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        for _, fname in pairs[self.max_save_files - 1:]:
            pathlib.Path(fname).unlink()

        if self.target_net is not None:
            target_net = self.target_net.state_dict()
            replay_buffer = self.replay_buffer.get_buffer()
        else:
            target_net = None
            replay_buffer = None
        save_objs = [self.model.state_dict(), target_net, self.optimizer.state_dict(), self.train_steps, replay_buffer]
        torch.save(save_objs, save_fname)

        if self.adversarial:
            save_objs = [self.dnet.state_dict(), self.adversarial_optim.state_dict()]
            torch.save(save_objs, save_fname+"disc")

        print("* Saved model to", save_fname)
