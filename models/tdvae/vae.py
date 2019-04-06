"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import glob
import pathlib

from pylego import ops

from ..basetdvae import BaseTDVAE
from ..gymtdvae import BaseGymTDVAE
from .discriminator import Discriminator
from .tdvae import ConvPreProcess, DBlock
from .adversarial import SAGANGenerator as ConvDecoder


class TDVAE(nn.Module):
    """ The full TD-VAE model with jumpy prediction.
    """

    def __init__(self,
                 x_size,
                 resnet_hidden_size,
                 processed_x_size,
                 b_size,
                 z_size,
                 layers,
                 samples_per_seq,
                 t_diff_min,
                 t_diff_max,
                 action_space=0,
                 action_dim=8):
        super().__init__()
        self.layers = layers
        self.samples_per_seq = samples_per_seq
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max

        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        # Input pre-process layer
        # Currently hard-coded for downsized Atari
        strides = [2, 2, 2, 2]
        scales = [2, 2, 2]
        l_per_block = [2, 2, 2, 2]
        self.process_x = ConvPreProcess(x_size, resnet_hidden_size, processed_x_size, scale=scales, stride=strides, l_per_block=l_per_block)

        # Multilayer state model is used. Sampling is done by sampling higher layers first.
        self.z_b = nn.ModuleList([DBlock(processed_x_size + (z_size if layer < layers - 1 else 0), b_size, z_size)
                                  for layer in range(layers)])

        # state to observation
        self.x_z = ConvDecoder(image_size=x_size, z_dim=layers * z_size, d_hidden=resnet_hidden_size)

        self.action_embedding = nn.Embedding(action_space, action_dim)

        self.time_encoding = torch.zeros(t_diff_max, t_diff_max)
        for i in range(t_diff_max):
            self.time_encoding[i, :i+1] = 1
        self.time_encoding = nn.Parameter(self.time_encoding.float(), requires_grad=False)

    def forward(self, x, actions):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)

        # q_B(z2 | b2)
        qb_z2_b2_mus, qb_z2_b2_logvars, qb_z2_b2s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](processed_x)
            else:
                qb_z2_b2_mu, qb_z2_b2_logvar = self.z_b[layer](torch.cat([processed_x, qb_z2_b2], dim=-1))
            qb_z2_b2_mus.insert(0, qb_z2_b2_mu)
            qb_z2_b2_logvars.insert(0, qb_z2_b2_logvar)

            qb_z2_b2 = ops.reparameterize_gaussian(qb_z2_b2_mu, qb_z2_b2_logvar, self.training)
            qb_z2_b2s.insert(0, qb_z2_b2)

        qb_z2_b2_mu = torch.cat(qb_z2_b2_mus, dim=1)
        qb_z2_b2_logvar = torch.cat(qb_z2_b2_logvars, dim=1)
        qb_z2_b2 = torch.cat(qb_z2_b2s, dim=1)


        # p_D(x2 | z2)
        pd_x2_z2 = self.x_z(qb_z2_b2.view(im_x.shape[0], -1))
        return (x, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2, pd_x2_z2)

    def visualize(self, x, t, n, actions):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        # aggregate the belief b
        # compute z from b
        p_z_bs = []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](processed_x)
            else:
                p_z_b_mu, p_z_b_logvar = self.z_b[layer](torch.cat([processed_x, p_z_b_mu], dim=-1))
            p_z_bs.insert(0, p_z_b_mu)

        z = torch.cat(p_z_bs, dim=1)
        rollout_x = []
        rollout_x.append(self.x_z(z.view(im_x.shape[0], -1)))
        return torch.stack(rollout_x, dim=1)



class GymVAE(BaseGymTDVAE):

    def __init__(self, flags, *args, **kwargs):
        self.device = torch.device("cuda" if flags.cuda else "cpu")
        model = TDVAE((3, 112, 80), flags.h_size, 64, flags.b_size, flags.z_size, flags.layers, flags.samples_per_seq,
                               flags.t_diff_min, flags.t_diff_max, action_space=20)
        if flags.adversarial:
            self.dnet = Discriminator(disc_size=flags.d_size, channels=3)
            self.adversarial_optim = torch.optim.Adam(self.dnet.parameters(), lr=flags.d_lr, betas=(0.0, 0.9))
            self.optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate, betas=(0.0, 0.9))
            flags.optimizer = self.optimizer
        self.adversarial = flags.adversarial
        self.beta = flags.beta
        self.d_weight = flags.d_weight
        self.d_steps = flags.d_steps

        super().__init__(model, flags, *args, **kwargs)

        if self.adversarial:
            self.dnet.to(self.device)
            self.model.to(self.device)

            for state in self.adversarial_optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def loss_function(self, forward_ret, labels=None, train=True):
        (x, qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2, pd_x2_z2) = forward_ret

        # replicate x multiple times
        x_flat = x.flatten(2, -1)
        x_flat = x_flat.expand(self.flags.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
        batch_size = x.size(0)

        if self.adversarial and self.model.training:
            r_in = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            f_in = pd_x2_z2.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            for i in range(self.d_steps):
                d_loss, g_loss, hidden_loss = self.dnet.get_loss(r_in, f_in)
                d_loss.backward(retain_graph=True)
                # print(d_loss, g_loss)
                self.adversarial_optim.step()
                self.adversarial_optim.zero_grad()
        else:
            g_loss = 0
            hidden_loss = 0

        eye = torch.ones(qb_z2_b2.size(-1)).to(qb_z2_b2.device)[None, None, :].expand(-1, qb_z2_b2.size(-2), -1)
        kl_div_qs_pb = ops.kl_div_gaussian(qb_z2_b2_mu,
                                           qb_z2_b2_logvar,
                                           0,
                                           eye).mean()

        target = x.flatten()
        pred = pd_x2_z2.flatten()
        bce = F.binary_cross_entropy(pred, target, reduction='sum') / batch_size
        bce_optimal = F.binary_cross_entropy(target, target, reduction='sum') / batch_size
        bce_diff = bce - bce_optimal

        if self.adversarial and self.model.training:
            r_in = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            f_in = pd_x2_z2.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            for i in range(self.d_steps):
                d_loss, g_loss, hidden_loss = self.dnet.get_loss(r_in, f_in)
                d_loss.backward(retain_graph=True)
                # print(d_loss, g_loss)
                self.adversarial_optim.step()
                self.adversarial_optim.zero_grad()
            bce_diff = hidden_loss
        else:
            g_loss = 0
            hidden_loss = 0

        loss = bce_diff + hidden_loss + self.d_weight*g_loss +\
               self.beta*kl_div_qs_pb

        return loss, bce_diff, kl_div_qs_pb, kl_div_qs_pb, bce_optimal
