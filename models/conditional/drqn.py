import collections
import glob
import gzip
import pathlib

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from pylego import ops

from ..baseconditional import BaseGymTDVAE


class PreProcess(nn.Module):
    """ The default preprocessing layer for 1d inputs.
    """

    def __init__(self, input_size, processed_x_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, processed_x_size*4)
        self.fc2 = nn.Linear(processed_x_size*4, processed_x_size)

    def forward(self, input_):
        t = F.elu(self.fc1(input_))
        t = F.elu(self.fc2(t))
        return t


class ConvPreProcess(nn.Module):
    """ The pre-process layer for image.
    """

    def __init__(self, input_size, d_hidden, d_out, blocks=None, scale=(2, 2), stride=(2, 2, 2),
                 l_per_block=(2, 2, 2)):
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
        x1 = F.elu(x1)
        x1 = self.bn1(x1)
        x2 = self.resnet(x1)
        x3 = self.fc1(x2.flatten(1, -1))
        x3 = F.elu(x3)
        x3 = self.bn2(x3)
        x4 = self.fc2(x3)
        return x4


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


class DRQN(nn.Module):

    def __init__(self, x_size, resnet_hidden_size, processed_x_size, b_size, layers, samples_per_seq,
                 action_space=0, action_dim=8, rl=False):
        super().__init__()
        self.layers = layers
        self.samples_per_seq = samples_per_seq

        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size

        self.rl = rl

        # Input pre-process layer
        if len(x_size) > 1:
            self.process_x = ConvPreProcess(x_size, resnet_hidden_size, processed_x_size)
        else:
            self.process_x = PreProcess(x_size, processed_x_size)

        # Multilayer LSTM for aggregating belief states
        self.b_rnn = ops.MultilayerLSTM(input_size=processed_x_size+action_dim+1, hidden_size=b_size, layers=layers,
                                        every_layer_input=True, use_previous_higher=True)

        # state to Q value per action
        if rl:
            self.q_z = QNetwork(layers * b_size, layers * b_size, action_space)

        self.action_embedding = nn.Embedding(action_space, action_dim)

    def compute_q(self, x, actions, rewards, done):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        if actions is not None:
            rewards = (rewards[..., None] / 10.0).clamp(-1.0, 1.0)
            action_embs = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, action_embs, rewards], -1)

        # aggregate the belief b
        b = self.b_rnn(processed_x, done)  # size: bs, time, layers, dim
        b = b[:, -1]  # size: bs, layers, dim
        z = b.view(b.size(0), -1)
        return self.q_z(z)

    def q_and_z_b(self, x, actions, rewards, done, t1, t2):
        # pre-process image x
        im_x = x.view(-1, self.x_size[0], self.x_size[1], self.x_size[2])
        processed_x = self.process_x(im_x)  # max x length is max(t2) + 1
        processed_x = processed_x.view(x.shape[0], x.shape[1], -1)
        if actions is not None:
            rewards = (rewards[..., None] / 10.0).clamp(-1.0, 1.0)
            action_embs = self.action_embedding(actions)
            processed_x = torch.cat([processed_x, action_embs, rewards], -1)
        else:
            action_embs = None

        # aggregate the belief b
        b = self.b_rnn(processed_x, done)  # size: bs, time, layers, dim

        # replicate b multiple times
        b = b[None, ...].expand(self.samples_per_seq, -1, -1, -1, -1)  # size: copy, bs, time, layers, dim

        # Element-wise indexing. sizes: bs, layers, dim
        b1 = torch.gather(b, 2, t1[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))
        b2 = torch.gather(b, 2, t2[..., None, None, None].expand(-1, -1, -1, b.size(3), b.size(4))).view(
            -1, b.size(3), b.size(4))

        z1 = b1.view(b1.size(0), -1)
        z2 = b2.view(b2.size(0), -1)
        if self.rl:
            # Q values
            q1 = self.q_z(z1)
            q2 = self.q_z(z2)
        else:
            q1, q2 = 0, 0

        return q1, q2

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

        q1, q2 = self.q_and_z_b(x, actions, rewards, done, t1, t2)
        return (x, actions, rewards, done, t1, t2, q1, q2)

    def visualize(self, *args):
        return None


class GymDRQN(BaseGymTDVAE):

    def __init__(self, flags, model=None, action_space=20, rl=False, replay_buffer=None, *args, **kwargs):
        self.rl = rl
        self.replay_buffer = replay_buffer

        if model is None:
            model_args = [(1, 40, 40), flags.h_size, 2*flags.b_size, flags.b_size, flags.layers, flags.samples_per_seq]
            model_kwargs = {'action_space': action_space}
            if rl:
                model_kwargs['rl'] = True
            model = DRQN(*model_args, **model_kwargs)
        super().__init__(model, flags, *args, **kwargs)

        if rl:
            self.target_net = DRQN(*model_args, **model_kwargs)
            self.target_net.eval()
            self.target_net.to(self.device)
        else:
            self.target_net = None

        if flags.load_file:
            self.load(flags.load_file)

    def update_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def loss_function(self, forward_ret, labels=None, loss=F.binary_cross_entropy):
        x_orig, actions, rewards, done, t1, t2, q1, q2 = forward_ret

        if self.rl:
            # Note: x[t], rewards[t] is a result of actions[t]
            # Q(s[t], a[t+1]) = r[t+1] + γ max_a Q(s[t+1], a)
            _, is_weight = labels

            # reward clipping for Atari
            clipped_rewards = rewards.clamp(-1.0, 1.0)

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

            actions = actions[None, ...].expand(self.flags.samples_per_seq, -1, -1)  # size: copy, bs, time
            a1_next = torch.gather(actions, 2, t1_next[..., None]).view(-1)  # size: bs
            a2_next = torch.gather(actions, 2, t2_next[..., None]).view(-1)  # size: bs

            pred_q1 = torch.gather(q1, 1, a1_next[..., None]).view(-1)
            pred_q2 = torch.gather(q2, 1, a2_next[..., None]).view(-1)

            q1_next = torch.gather(q1_next_target, 1, q1_next_index).view(-1)
            q2_next = torch.gather(q2_next_target, 1, q2_next_index).view(-1)
            target_q1 = r1_next + self.flags.discount_factor * (1.0 - done1_next) * q1_next
            target_q2 = r2_next + self.flags.discount_factor * (1.0 - done2_next) * q2_next

            rl_loss = 0.5 * (F.smooth_l1_loss(pred_q1, target_q1, reduction='none') +
                             F.smooth_l1_loss(pred_q2, target_q2, reduction='none'))
            # errors for prioritized experience replay
            rl_errors = 0.5 * (torch.abs(pred_q1 - target_q1) + torch.abs(pred_q2 - target_q2)).detach()
        else:
            rl_loss = 0.0
            is_weight = 1.0
            rl_errors = 0.0

        # multiply is_weight separately for ease of reporting
        rl_loss = is_weight * rl_loss
        loss = self.flags.rl_weight * rl_loss

        if self.rl:  # workaround to work with non-RL setting
            rl_loss = rl_loss.mean()
        return collections.OrderedDict([('loss', loss.mean()),
                                        ('bce_diff', 0),
                                        ('returns_loss', 0),
                                        ('kl_div_qs_pb', 0),
                                        ('kl_shift_qb_pt', 0),
                                        ('rl_loss', rl_loss),
                                        ('bce_optimal', 0),
                                        ('rl_errors', rl_errors)])

    def initialize(self, load_file):
        '''Overriding: do not load file during superclass initialization, we do it manually later in init'''
        pass

    def load(self, load_file):
        """Load a model from a saved file."""
        print("* Loading model from", load_file, "...")
        with gzip.open(load_file, 'rb', compresslevel=1) as f:
            m_state_dict, target_net_state_dict, o_state_dict, train_steps, replay_buffer = torch.load(f)
        self.model.load_state_dict(m_state_dict)
        if self.target_net is not None and target_net_state_dict is not None:
            self.target_net.load_state_dict(target_net_state_dict)
            self.replay_buffer.load_buffer(replay_buffer)
        self.optimizer.load_state_dict(o_state_dict)
        self.train_steps = train_steps
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
        with gzip.open(save_fname, 'wb', compresslevel=1) as f:
            torch.save(save_objs, f)
        print("* Saved model to", save_fname)
