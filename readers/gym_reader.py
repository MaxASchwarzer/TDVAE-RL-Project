"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""

import numpy as np
import torch as T
from torch.utils import data
from torchvision import datasets
import torch.nn.functional as F
import gym
import gym_vecenv as vecenv
from collections import deque
from pylego.reader import Reader


class GymReader(Reader):

    def __init__(self,
                env,
                batch_size,
                seq_len,
                iters_per_epoch,
                done_policy=np.any):

        fn = lambda : gym.make(env)
        env_fcns = [fn for i in range(batch_size)]
        self.batch_size = batch_size
        self.env_name = env
        self.env = vecenv.SubprocVecEnv(env_fcns)
        self.env.reset()
        self.seq_len = seq_len
        self.sample_env = gym.make(env)
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.done = False
        self.buffers = [deque(maxlen=seq_len) for i in range(5)]
        self.done_policy = done_policy

        super().__init__({'train': iters_per_epoch, 'val': iters_per_epoch//10, 'test': iters_per_epoch//10})

    def reset(self):
        self.iters = 0
        self.done = False
        self.env.reset()

    def iter_batches(self,
                     split_name,
                     batch_size,
                     actions=None,
                     shuffle=True,
                     partial_batching=False,
                     threads=1,
                     epochs=1,
                     fill_buffer=True,
                     max_batches=-1,
                     inner_frameskip=10):

        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            actions = [self.sample_env.action_space.sample() for i in range(self.batch_size)]
        for i in range(inner_frameskip):
            self.env.step_async(actions)
            obs, rewards, done, meta = self.env.step_wait()
            obs = np.transpose(obs, axes=(0, 3, 1, 2))
            obs = T.tensor(obs).float()/256
            if self.env_name == "Pong-v0":
                obs = F.pad(obs, (0, 0, 0, 14), mode="constant", value=0)


            for i, val in enumerate([obs, actions, rewards, done, meta]):
                self.buffers[i].append(val)
            if self.done_policy(done):
                self.env.reset()
                for buffer in self.buffers:
                  buffer.clear()

        if fill_buffer and len(self.buffers[0]) < self.seq_len:
            return self.iter_batches(split_name, batch_size, actions=actions, shuffle=shuffle)

        output = [T.stack(list(self.buffers[0]), 1)] + [np.array(buffer).swapaxes(0, 1) for buffer in self.buffers[1:]]

        output = [o[:batch_size] for o in output]
        self.iters += 1
        if self.iters > self.iters_per_epoch:
            self.done = True
        return output
