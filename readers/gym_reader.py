from collections import deque
from multiprocessing import Process, Pipe
import pickle

import gym
import gym_vecenv as vecenv
import numpy as np
import torch
import torch.nn.functional as F

from pylego.reader import Reader


def make_env(env_name, frameskip=3, steps=1000000, secs=100000):
    env = gym.make(env_name)
    env.env.frameskip = frameskip
    env._max_episode_steps = steps
    env._max_episode_seconds = secs
    return env


class ActionConditionalBatch:  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, downsample=True, inner_frameskip=3, data_dir='data'):
        self.env_name = env
        self.seq_len = seq_len
        self.batch_size = batch_size

        fn = lambda: make_env(env, frameskip=inner_frameskip)
        env_fcns = [fn for i in range(batch_size)]
        self.env = SubprocVecEnv(env_fcns, n_workers=threads)
        self.env.reset()
        self.buffers = [deque(maxlen=seq_len) for i in range(6)]
        self.downsample = downsample

        sample_env = gym.make(env)
        self.action_space = sample_env.action_space.n
        sample_env.close()

        with open(data_dir + '/' + env + '/img_stats.pk', 'rb') as f:
            (self.img_mean, self.img_std, self.img_min, self.img_max, self.img_true_min, self.img_true_max,
             self.img_hcrop_top, self.img_hcrop_bottom, self.img_vcrop_left, self.img_vcrop_right) = pickle.load(f)
            self.img_mean = torch.tensor(self.img_mean)
            self.img_std = torch.tensor(self.img_std)

    def get_next(self, actions=None, fill_buffer=True, outer_frameskip=2):
        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            in_actions = np.random.randint(0, self.action_space, size=self.batch_size)
        else:
            in_actions = actions

        for _ in range(outer_frameskip):
            self.env.step_async(in_actions)
            orig_obs, rewards, done, meta = self.env.step_wait()
            orig_obs = np.transpose(orig_obs, axes=(0, 3, 1, 2))  # original unnormalized images
            obs = torch.tensor(orig_obs).float() / 255
            if "Pong" in self.env_name or "Seaquest" in self.env_name and self.downsample:
                obs = F.pad(obs, (0, 0, 0, 14), mode="constant", value=0)
                obs = F.avg_pool2d(obs, (2, 2,), stride=2)

            obs = ((obs - self.img_mean) / self.img_std).clamp_(self.img_min, self.img_max)
            obs = (obs - self.img_min) / (self.img_max - self.img_min)

            # FIXME crop images instead of filling zeros, this is wasting compute
            if self.img_hcrop_top > 0:
                obs[:, :, :self.img_hcrop_top].fill_(0.5)
            if self.img_hcrop_bottom > 0:
                obs[:, :, -self.img_hcrop_bottom:].fill_(0.5)
            if self.img_vcrop_left > 0:
                obs[:, :, :, :self.img_vcrop_left].fill_(0.5)
            if self.img_vcrop_right > 0:
                obs[:, :, :, -self.img_vcrop_right:].fill_(0.5)

            for i, val in enumerate([obs, in_actions, rewards, done, meta, orig_obs]):
                self.buffers[i].append(val)

        if fill_buffer and len(self.buffers[0]) < self.seq_len:
            return self.get_next(actions=actions, fill_buffer=True, outer_frameskip=outer_frameskip)

        output = [torch.stack(list(self.buffers[0]), 1)] + [np.array(buffer).swapaxes(0, 1)
                                                            for buffer in self.buffers[1:]]

        output = [o[:self.batch_size] for o in output]
        return output

    def close(self):
        self.env.close()


class GymReader(Reader):  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, iters_per_epoch):
        self.env = env
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.threads = threads
        super().__init__({'train': iters_per_epoch})
        self.action_conditional_batch = ActionConditionalBatch(env, seq_len, batch_size, threads)

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


class ReplayBuffer(Reader):  # TODO prioritized experience replay

    def __init__(self, emulator, buffer_size, iters_per_epoch, skip_init=False):
        self.buffer = deque(maxlen=buffer_size)
        if skip_init:
            print('* Skipping replay buffer initialization')
        else:
            print('* Initializing replay buffer')
            while len(self.buffer) < buffer_size:
                for conditional_batch in emulator.iter_batches('train', emulator.batch_size, threads=emulator.threads,
                                                               max_batches=int(np.ceil(buffer_size /
                                                                                       emulator.batch_size))):
                    self.add(conditional_batch.get_next()[:3])
                    if len(self.buffer) >= buffer_size:
                        break
            print('* Replay buffer initialized')
        super().__init__({'train': iters_per_epoch})

    def add(self, trajs):
        obs, actions, rewards = trajs
        self.buffer.extend(zip(obs, actions, rewards))

    def get_buffer(self):
        return self.buffer

    def load_buffer(self, buffer):
        print('* Loading external replay buffer')
        self.buffer = buffer

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        assert split_name == 'train'
        assert shuffle

        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        for _ in range(epochs * epoch_size):
            # specify p here to change uniform sampling:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            batch = list(zip(*(self.buffer[i] for i in indices)))
            yield torch.stack(batch[0]), np.array(batch[1]), np.array(batch[2])


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
            remote.send((np.array(obs, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones),
                         np.array(infos)))
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
