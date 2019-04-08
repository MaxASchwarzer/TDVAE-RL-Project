import glob
import pathlib

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from pylego import ops

from ..baseconditional import BaseGymTDVAE
from .utils import Discriminator, SAGANGenerator


class DBlock(nn.Module):
    """ A basic building block for computing parameters of a normal distribution.
    Corresponds to D in the appendix."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        t = torch.tanh(self.fc1(input_))
        t = t * torch.sigmoid(self.fc2(input_))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class PreProcess(nn.Module):
    """ The pre-process layer for MNIST image.
    """

    def __init__(self, input_size, processed_x_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input_):
        t = torch.relu(self.fc1(input_))
        t = torch.relu(self.fc2(t))
        return t


class ConvPreProcess(nn.Module):
    """ The pre-process layer for image.
    """

    def __init__(self,
                 input_size,
                 d_hidden,
                 d_out,
                 blocks=None,
                 scale=(2, 2, 2),
                 stride=(2, 2, 2, 2),
                 l_per_block=(2, 2, 2, 2)):
        super().__init__()

        d_in = input_size[0]
        input_size = input_size[1:]

        self.initial = nn.Conv2d(d_in, d_hidden, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(d_hidden)

        if blocks is None:
            scales = np.cumprod([1] + list(scale))
            blocks = [(l_per_block, int(d_hidden*scale), stride) for l_per_block, scale, stride in
                      zip(l_per_block, scales, stride)]

        self.resnet = ops.ResNet(d_hidden, blocks) # Will by default downscale to 224//16, 160//16,

        self.final_shape = [int(i//(np.prod(stride))) for i in input_size]
        self.total_size = np.prod(self.final_shape)*blocks[-1][1]

        self.fc1 = nn.Linear(self.total_size, d_out*4)
        self.bn2 = nn.BatchNorm1d(d_out*4)
        self.fc2 = nn.Linear(d_out*4, d_out)

    def forward(self, x):
        x1 = self.initial(x)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)
        x2 = self.resnet(x1)
        x3 = self.fc1(x2.flatten(1, -1))
        x3 = F.relu(x3)
        x3 = self.bn2(x3)
        x4 = self.fc2(x3)
        return x4


class ConvDecoder(nn.Module):
    """ The pre-process layer for image.
    """

    def __init__(self,
                 d_out,
                 d_hidden,
                 input_size,
                 blocks=None,
                 scale=(2, 2, 2, 2),
                 stride=(2, 2, 2, 2, 2),
                 l_per_block=(4, 4, 4, 4, 4)):
        super().__init__()

        d_in = input_size[0]
        input_size = input_size[1:]

        if blocks is None:
            scales = np.cumprod([1] + list(scale))
            blocks = [(l_per_block, int(d_hidden*scale), -stride) for l_per_block, scale, stride in
                      zip(l_per_block, scales, stride)]

        print(blocks)

        self.final_shape = [int(i//np.prod(stride)) for i in input_size]
        self.biases = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.total_size = int(np.prod(self.final_shape)*blocks[-1][1])
        print(self.final_shape, self.total_size)
        self.fc1 = nn.Linear(d_out, d_out*4)
        self.bn1 = nn.BatchNorm1d(d_out*4)
        self.fc2 = nn.Linear(d_out*4, self.total_size)
        self.bn2 = nn.BatchNorm2d(blocks[-1][1])

        self.resnet = ops.ResNet(blocks[-1][1], list(reversed(blocks)), skip_last_norm=False)

        self.final = nn.Conv2d(d_hidden, d_in, 7, 1, 3)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)
        x2 = self.fc2(x1)
        x2 = F.relu(x2)
        x2 = x2.view(x.shape[0], x2.shape[1]//np.prod(self.final_shape), self.final_shape[0], self.final_shape[1])
        x2 = self.bn2(x2)
        x3 = self.resnet(x2)
        x4 = self.final(x3)
        return torch.sigmoid(x4.flatten(1, -1))


class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    """

    def __init__(self, z_size, hidden_size, x_size):
        super().__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p


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
        self.process_x = ConvPreProcess(x_size, resnet_hidden_size, processed_x_size)

        # Multilayer LSTM for aggregating belief states
        self.b_rnn = ops.MultilayerLSTM(input_size=processed_x_size+action_dim, hidden_size=b_size, layers=layers,
                                        every_layer_input=True, use_previous_higher=True)

        # Multilayer state model is used. Sampling is done by sampling higher layers first.
        self.z_b = nn.ModuleList([DBlock(b_size + (z_size if layer < layers - 1 else 0), 50, z_size)
                                  for layer in range(layers)])

        # Given belief and state at time t2, infer the state at time t1
        self.z1_z2_b = nn.ModuleList([DBlock(b_size + layers * z_size + action_dim +
                                             (z_size if layer < layers - 1 else 0) + t_diff_max,
                                             50, z_size)
                                      for layer in range(layers)])

        # Given the state at time t1, model state at time t2 through state transition
        self.z2_z1 = nn.ModuleList([DBlock(layers * z_size + action_dim +
                                           (z_size if layer < layers - 1 else 0) + t_diff_max,
                                           50, z_size)
                                    for layer in range(layers)])

        # state to observation
        # self.x_z = ConvDecoder(layers * z_size, resnet_hidden_size, x_size)
        self.x_z = SAGANGenerator(x_size, z_dim=layers * z_size, d_hidden=resnet_hidden_size)

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
        if actions is not None:
            actions = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, actions], -1)

        # aggregate the belief b
        b = self.b_rnn(processed_x)  # size: bs, time, layers, dim

        # replicate b multiple times
        b = b[None, ...].expand(self.samples_per_seq, -1, -1, -1, -1)  # size: copy, bs, time, layers, dim

        t1 = torch.randint(0, x.size(1) - self.t_diff_max, (b.size(0), b.size(1)), device=b.device)
        t2 = t1 + torch.randint(self.t_diff_min, self.t_diff_max + 1, (b.size(0), b.size(1)), device=b.device)
        t_encodings = self.time_encoding[t2 - t1 - 1].reshape(-1, self.t_diff_max).contiguous()

        # Element-wise indexing. sizes: bs, layers, dim
        b1 = torch.gather(b, 2, t1[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))
        b2 = torch.gather(b, 2, t2[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))
        if actions is not None:
            actions = actions[None, ...].expand(self.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
            a = torch.gather(actions, 2, t1[..., None, None].expand(-1, -1, -1,
                                                                    actions.shape[-1])).view(-1, actions.shape[-1])
            # b1 = torch.cat([b1, a[:, None, :].expand(-1, self.layers, -1)], -1)

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

        # q_S(z1 | z2, b1, b2) ~= q_S(z1 | z2, b1)
        qs_z1_z2_b1_mus, qs_z1_z2_b1_logvars, qs_z1_z2_b1s = [], [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b[layer](torch.cat([qb_z2_b2, b1[:, layer],
                                                                                    t_encodings, a], dim=1))
            else:
                qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar = self.z1_z2_b[layer](torch.cat([qb_z2_b2, b1[:, layer],
                                                                                    qs_z1_z2_b1, t_encodings, a],
                                                                                   dim=1))
            qs_z1_z2_b1_mus.insert(0, qs_z1_z2_b1_mu)
            qs_z1_z2_b1_logvars.insert(0, qs_z1_z2_b1_logvar)

            qs_z1_z2_b1 = ops.reparameterize_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, self.training)
            qs_z1_z2_b1s.insert(0, qs_z1_z2_b1)

        qs_z1_z2_b1_mu = torch.cat(qs_z1_z2_b1_mus, dim=1)
        qs_z1_z2_b1_logvar = torch.cat(qs_z1_z2_b1_logvars, dim=1)
        qs_z1_z2_b1 = torch.cat(qs_z1_z2_b1s, dim=1)

        # p_T(z2 | z1), also conditions on q_B(z2) from higher layer
        pt_z2_z1_mus, pt_z2_z1_logvars = [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_z2_b1, t_encodings, a], dim=1))
            else:
                pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](torch.cat([qs_z1_z2_b1,
                                                                            qb_z2_b2s[layer + 1],
                                                                            t_encodings, a], dim=1))
            pt_z2_z1_mus.insert(0, pt_z2_z1_mu)
            pt_z2_z1_logvars.insert(0, pt_z2_z1_logvar)

        pt_z2_z1_mu = torch.cat(pt_z2_z1_mus, dim=1)
        pt_z2_z1_logvar = torch.cat(pt_z2_z1_logvars, dim=1)

        # p_B(z1 | b1)
        pb_z1_b1_mus, pb_z1_b1_logvars = [], []
        for layer in range(self.layers - 1, -1, -1):
            if layer == self.layers - 1:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](b1[:, layer])
            else:
                pb_z1_b1_mu, pb_z1_b1_logvar = self.z_b[layer](torch.cat([b1[:, layer],
                                                                          qs_z1_z2_b1s[layer + 1]],
                                                                         dim=1))
            pb_z1_b1_mus.insert(0, pb_z1_b1_mu)
            pb_z1_b1_logvars.insert(0, pb_z1_b1_logvar)

        pb_z1_b1_mu = torch.cat(pb_z1_b1_mus, dim=1)
        pb_z1_b1_logvar = torch.cat(pb_z1_b1_logvars, dim=1)

        # p_D(x2 | z2)
        pd_x2_z2 = self.x_z(qb_z2_b2)

        return (x, t2, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar,
                qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2)

    def visualize(self, x, t, n, actions):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        if actions is not None:
            actions = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, actions], -1)
        # aggregate the belief b
        b = self.b_rnn(processed_x)[:, t]  # size: bs, time, layers, dim
        t_encodings = self.time_encoding[0].reshape(-1, self.t_diff_max).contiguous()
        t_encodings = t_encodings.expand(b.shape[0], -1)

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
                        inputs = torch.cat([z, t_encodings, actions[:, i+1]], dim=1)
                    else:
                        inputs = torch.cat([z, t_encodings], dim=1)
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](inputs)
                else:
                    if actions is not None:
                        inputs = torch.cat([z, pt_z2_z1, t_encodings, actions[:, i+1]], dim=1)
                    else:
                        inputs = torch.cat([z, pt_z2_z1, t_encodings], dim=1)
                    pt_z2_z1_mu, pt_z2_z1_logvar = self.z2_z1[layer](inputs)
                pt_z2_z1 = ops.reparameterize_gaussian(pt_z2_z1_mu, pt_z2_z1_logvar, True)
                next_z.insert(0, pt_z2_z1_mu)

            z = torch.cat(next_z, dim=1)
            rollout_x.append(self.x_z(z))

        return torch.stack(rollout_x, dim=1)


class GymTDVAE(BaseGymTDVAE):

    def __init__(self, flags, model=None, *args, **kwargs):
        self.adversarial = flags.adversarial
        self.beta = flags.beta
        self.d_weight = flags.d_weight
        self.d_steps = flags.d_steps

        if model is None:
            model = TDVAE((3, 112, 80), flags.h_size, 64, flags.b_size, flags.z_size, flags.layers,
                          flags.samples_per_seq, flags.t_diff_min, flags.t_diff_max, action_space=20)
        if self.adversarial:
            kwargs['optimizer'] = None  # we create an optimizer later
        super().__init__(model, flags, *args, **kwargs)

        if self.adversarial:
            self.dnet = Discriminator(channels=3, d_size=flags.d_size)
            self.dnet.to(self.device)

            self.adversarial_optim = torch.optim.Adam(self.dnet.parameters(), lr=flags.d_lr, betas=(0.0, 0.9))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=flags.learning_rate, betas=(0.0, 0.9))
            flags.optimizer = self.optimizer

        if flags.load_file:
            self.load(flags.load_file)

    def loss_function(self, forward_ret, labels=None):
        (x, t2, qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar, qb_z2_b2_mu, qb_z2_b2_logvar,
         qb_z2_b2, pt_z2_z1_mu, pt_z2_z1_logvar, pd_x2_z2) = forward_ret

        # replicate x multiple times
        x_flat = x.flatten(2, -1)
        x_flat = x_flat.expand(self.flags.samples_per_seq, -1, -1, -1)  # size: copy, bs, time, dim
        x2 = torch.gather(x_flat, 2, t2[..., None, None].expand(-1, -1, -1, x_flat.size(3))).view(-1, x_flat.size(3))
        batch_size = x2.size(0)
        kl_div_qs_pb = ops.kl_div_gaussian(qs_z1_z2_b1_mu, qs_z1_z2_b1_logvar, pb_z1_b1_mu, pb_z1_b1_logvar).mean()

        kl_shift_qb_pt = (ops.gaussian_log_prob(qb_z2_b2_mu, qb_z2_b2_logvar, qb_z2_b2) -
                          ops.gaussian_log_prob(pt_z2_z1_mu, pt_z2_z1_logvar, qb_z2_b2)).mean()

        bce = F.binary_cross_entropy(pd_x2_z2, x2, reduction='sum') / batch_size
        bce_optimal = F.binary_cross_entropy(x2, x2, reduction='sum') / batch_size
        bce_diff = bce - bce_optimal

        if self.adversarial and self.is_training():
            r_in = x2.view(x2.shape[0], x.shape[2], x.shape[3], x.shape[4])
            f_in = pd_x2_z2.view(x2.shape[0], x.shape[2], x.shape[3], x.shape[4])
            for _ in range(self.d_steps):
                d_loss, g_loss, hidden_loss = self.dnet.get_loss(r_in, f_in)
                d_loss.backward(retain_graph=True)
                # print(d_loss, g_loss)
                self.adversarial_optim.step()
                self.adversarial_optim.zero_grad()
            bce_diff = hidden_loss  # XXX bce_diff added twice to loss?
        else:
            g_loss = 0
            hidden_loss = 0

        loss = bce_diff + hidden_loss + self.d_weight * g_loss + self.beta * (kl_div_qs_pb + kl_shift_qb_pt)

        return loss, bce_diff, kl_div_qs_pb, kl_shift_qb_pt, bce_optimal

    def initialize(self, load_file):
        '''Overriding: do not load file during superclass initialization, we do it manually later in init'''
        pass

    def load(self, load_file):
        """Load a model from a saved file."""
        print("* Loading model from", load_file, "...")
        m_state_dict, o_state_dict, train_steps = torch.load(load_file)
        self.model.load_state_dict(m_state_dict)
        self.optimizer.load_state_dict(o_state_dict)
        self.train_steps = train_steps
        if self.adversarial:
            m_state_dict, o_state_dict = torch.load(load_file+"disc")
            self.dnet.load_state_dict(m_state_dict)
            self.adversarial_optim.load_state_dict(o_state_dict)
        print("* Loaded model from", load_file)

    def save(self, save_file, max_files=5):
        "Save model to file."
        save_fname = save_file + "." + str(self.train_steps)
        print("* Saving model to", save_fname, "...")
        existing = glob.glob(save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        for _, fname in pairs[max_files - 1:]:
            pathlib.Path(fname).unlink()

        save_objs = [self.model.state_dict(), self.optimizer.state_dict(), self.train_steps]
        torch.save(save_objs, save_fname)

        if self.adversarial:
            save_objs = [self.dnet.state_dict(), self.adversarial_optim.state_dict()]
            torch.save(save_objs, save_fname+"disc")

        print("* Saved model to", save_fname)