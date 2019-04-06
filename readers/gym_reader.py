from collections import deque
from multiprocessing import Process, Pipe

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

    def __init__(self, env, seq_len, batch_size, threads, shuffle, downsample=True, inner_frameskip=3):
        self.env_name = env
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        fn = lambda: make_env(env, frameskip=inner_frameskip)
        env_fcns = [fn for i in range(batch_size)]
        self.env = SubprocVecEnv(env_fcns, n_workers=threads)
        self.env.reset()
        self.sample_env = gym.make(env)
        self.buffers = [deque(maxlen=seq_len) for i in range(5)]
        self.downsample = downsample

    def get_next(self, actions=None, shuffle=True, fill_buffer=True, outer_frameskip=10):
        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            actions = [self.sample_env.action_space.sample() for i in range(self.batch_size)]

        for _ in range(outer_frameskip):
            self.env.step_async(actions)
            obs, rewards, done, meta = self.env.step_wait()
            obs = np.transpose(obs, axes=(0, 3, 1, 2))
            obs = torch.tensor(obs).float() / 256
            if "Pong" in self.env_name or "Seaquest" in self.env_name and self.downsample:
                obs = F.pad(obs, (0, 0, 0, 14), mode="constant", value=0)
                obs = F.avg_pool2d(obs, (2, 2,), stride=2)

            for i, val in enumerate([obs, actions, rewards, done, meta]):
                self.buffers[i].append(val)

        if fill_buffer and len(self.buffers[0]) < self.seq_len:
            return self.get_next(actions=actions, shuffle=shuffle, fill_buffer=True, outer_frameskip=outer_frameskip)

        output = [torch.stack(list(self.buffers[0]), 1)] + [np.array(buffer).swapaxes(0, 1)
                                                            for buffer in self.buffers[1:]]

        output = [o[:self.batch_size] for o in output]
        return output


class GymReader(Reader):  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, iters_per_epoch):
        self.env = env
        self.seq_len = seq_len
        super().__init__({'train': iters_per_epoch,
                          'val': int(iters_per_epoch * 0.15),
                          'test': int(iters_per_epoch * 0.15)})

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        action_conditional_batch = ActionConditionalBatch(self.env, self.seq_len, batch_size, threads, shuffle)
        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        for _ in range(epochs * epoch_size):
            yield action_conditional_batch


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
            remote.send((np.stack(obs, 0), np.stack(rewards, 0), np.stack(dones, 0), np.stack(infos, 0)))
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
