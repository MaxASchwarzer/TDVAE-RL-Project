from collections import deque
from multiprocessing import Process, Pipe
import pickle

import gym
import gym_vecenv as vecenv
import numpy as np
import torch
import torch.nn.functional as F

from pylego import misc
from pylego.reader import Reader


def make_env(env_name, frameskip=4, steps=1000000, secs=100000):
    env = gym.make(env_name)
    env.env.frameskip = frameskip
    env._max_episode_steps = steps
    env._max_episode_seconds = secs
    return env


class ActionConditionalBatch:  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, downsample=True, inner_frameskip=4, raw=False,
                 data_dir='data'):
        self.env_name = env
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.raw = raw  # return raw images

        fn = lambda: make_env(env, frameskip=inner_frameskip)
        env_fcns = [fn for i in range(batch_size)]
        self.env = SubprocVecEnv(env_fcns, n_workers=threads)
        self.env.reset()
        self.buffers = [deque(maxlen=seq_len) for i in range(6)]
        self.downsample = downsample

        sample_env = gym.make(env)
        self.action_space = sample_env.action_space.n
        sample_env.close()

        if not raw:
            with open(data_dir + '/' + env + '/img_stats.pk', 'rb') as f:
                (self.img_mean, self.img_std, self.img_min, self.img_max, self.img_true_min, self.img_true_max,
                 self.img_hcrop_top, self.img_hcrop_bottom, self.img_vcrop_left, self.img_vcrop_right) = pickle.load(f)
                self.img_mean = torch.tensor(self.img_mean)
                self.img_std = torch.tensor(self.img_std)

    def get_next(self, actions=None, fill_buffer=True):
        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            in_actions = np.random.randint(0, self.action_space, size=self.batch_size)
        else:
            in_actions = actions

        self.env.step_async(in_actions)
        orig_obs, rewards, done, meta = self.env.step_wait()
        orig_obs = np.transpose(orig_obs, axes=(0, 3, 1, 2))  # original unnormalized images
        obs = torch.tensor(orig_obs).float() / 255

        if not self.raw:
            obs = ((obs - self.img_mean) / self.img_std).clamp_(self.img_min, self.img_max)
            obs = (obs - self.img_min) / (self.img_max - self.img_min)

            h, w = obs.size()[2:]
            obs = obs[:, :, self.img_hcrop_top:h-self.img_hcrop_bottom, self.img_vcrop_left:w-self.img_vcrop_right]
            h, w = obs.size()[2:]
            pad_h, pad_w = 160 - h, 160 - w
            if self.downsample:
                obs = F.avg_pool2d(obs, (2, 2,), stride=2)
                pad_h = np.ceil(pad_h / 2.0)
                pad_w = np.ceil(pad_w / 2.0)
            pad_h /= 2.0
            pad_w /= 2.0

            obs = F.pad(obs, (int(np.ceil(pad_w)), int(np.floor(pad_w)), int(np.ceil(pad_h)), int(np.floor(pad_h))),
                        mode="constant", value=0.5)

        for i, val in enumerate([obs, in_actions, rewards, done, meta, orig_obs]):
            self.buffers[i].append(val)

        if fill_buffer and len(self.buffers[0]) < self.seq_len:
            return self.get_next(actions=actions, fill_buffer=True)

        output = [torch.stack(list(self.buffers[0]), 1)] + [np.array(buffer).swapaxes(0, 1)
                                                            for buffer in self.buffers[1:]]

        output = [o[:self.batch_size] for o in output]
        return output

    def close(self):
        self.env.close()


class GymReader(Reader):  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, iters_per_epoch, raw=False):
        self.env = env
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.threads = threads
        super().__init__({'train': iters_per_epoch})
        self.action_conditional_batch = ActionConditionalBatch(env, seq_len, batch_size, threads, raw=raw)

    def action_space(self):
        return self.action_conditional_batch.action_space

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        assert split_name == 'train'
        assert batch_size == self.batch_size
        assert threads == self.threads

        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        if epoch_size is np.inf:
            generator = iter(int, 1)
        else:
            generator = range(epochs * epoch_size)
        for _ in generator:
            yield self.action_conditional_batch

    def close(self):
        self.action_conditional_batch.close()


class ReplayBuffer(Reader):
    '''Replay buffer implementing prioritized experience replay.'''

    def __init__(self, emulator, buffer_size, iters_per_epoch, t_diff_min, t_diff_max, gamma, clip_errors=2.0,
                 skip_init=False):
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max
        self.clip_errors = clip_errors  # XXX default value ideal only for Seaquest
        self.buffer = misc.SumTree(buffer_size)
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.e = 0.01
        self.a = 0.6
        self.gammas = np.array([gamma ** i for i in range(emulator.seq_len)])

        if skip_init:
            print('* Skipping replay buffer initialization')
        else:
            print('* Initializing replay buffer')
            while self.buffer.count < buffer_size:
                for conditional_batch in emulator.iter_batches('train', emulator.batch_size, threads=emulator.threads,
                                                               max_batches=int(np.ceil(buffer_size /
                                                                                       emulator.batch_size))):
                    self.add(conditional_batch.get_next()[:4])
                    if self.buffer.count >= buffer_size:
                        break
            print('* Replay buffer initialized')
        super().__init__({'train': iters_per_epoch})

    def calc_priority(self, error):
        return np.minimum(self.clip_errors, error + self.e) ** self.a

    def add(self, trajs, t_diff_min=None, t_diff_max=None):
        t_diff_min = t_diff_min or self.t_diff_min
        t_diff_max = t_diff_max or self.t_diff_max

        priority = self.calc_priority(np.inf)
        for ob, action, reward, done in zip(*trajs):
            # generate random (t1, t2) combination
            # TODO don't let there be a done between t1 and t2
            t1 = np.random.randint(0, ob.size(0) - t_diff_max - 1)  # -1 to leave room for next reward
            t2 = t1 + np.random.randint(t_diff_min, t_diff_max + 1)
            if t1 + 1 <= t2:
                returns = (self.gammas[:t2 - t1] * reward[t1 + 1:t2 + 1]).sum()
            else:
                returns = 0.0
            self.buffer.add(priority, (ob, action, reward, done, t1, t2, returns))

    def update(self, indices, errors):
        priorities = self.calc_priority(errors)
        for idx, priority in zip(indices, priorities):
            self.buffer.update(idx, priority)

    def get_buffer(self):
        return (self.buffer, self.beta)

    def load_buffer(self, buffer):
        print('* Loading external replay buffer')
        self.buffer, self.beta = buffer

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.buffer.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.buffer.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.buffer.total()
        is_weight = np.power(self.buffer.count * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        assert split_name == 'train'
        assert shuffle

        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        for _ in range(epochs * epoch_size):
            batch, idxs, is_weight = self.sample(batch_size)
            obs, actions, rewards, done, t1, t2, returns = list(zip(*batch))
            yield (torch.stack(obs), np.array(actions), np.array(rewards), np.array(done), np.array(t1), np.array(t2),
                   np.array(returns), np.array(is_weight), idxs)


# The following code is largely adapted from https://github.com/agakshat/gym_vecenv,
# and carries the following license:
# MIT License
#
# Copyright (c) 2018 Akshat Agarwal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def worker(remote, parent_remote, env_fn_wrappers):
    parent_remote.close()
    envs = [env_fn_wrapper.x() for env_fn_wrapper in env_fn_wrappers]
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rewards, dones, infos = [], [], [], []
            for env, action in zip(envs, data):
                ob, reward, done, info = env.step(action)
                if np.any(done):
                    ob = env.reset()
                obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            remote.send((np.array(obs, dtype=np.float32), np.array(rewards, dtype=np.float32),
                         np.array(dones, dtype=np.float32), np.array(infos)))
        elif cmd == 'reset':
            ob = [env.reset() for env in envs]
            ob = np.stack(ob, 0)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = [env.reset_task() for env in envs]
            ob = np.stack(ob, 0)
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(([(env.observation_space, env.action_space) for env in envs]))
        else:
            raise NotImplementedError


class SubprocVecEnv(vecenv.vec_env.VecEnv):
    def __init__(self, env_fns, n_workers=-1, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.n_workers = n_workers
        if n_workers == -1:
            self.n_workes = nenvs
        env_fns = np.array_split(env_fns, n_workers)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_workers)])
        self.ps = [Process(target=worker,
                           args=(work_remote, remote, [vecenv.vec_env.CloudpickleWrapper(fn) for fn in env_fns]))
                   for (work_remote, remote, env_fns) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        spaces = self.remotes[0].recv()
        vecenv.vec_env.VecEnv.__init__(self, len(env_fns), spaces[0][0], spaces[0][1])

    def step_async(self, actions):
        actions = np.array_split(actions, self.n_workers)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs, 0), np.concatenate(rews, 0), np.concatenate(dones, 0), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.concatenate([remote.recv() for remote in self.remotes], 0)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.concatenate([remote.recv() for remote in self.remotes], 0)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
