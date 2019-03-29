"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
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

        self.envs = [gym.make(env) for i in range(batch_size)]
        self.env = vecenv.SubprocVecEnv(self.envs)
        self.buffers = [deque(maxlen=seq_len) for i in range(5)]
        self.done_policy = done_policy

        super().__init__({'train': iters_per_epoch, 'val': iters_per_epoch, 'test': iters_per_epoch})

    def iter_batches(self,
                     split_name,
                     batch_size,
                     actions=None,
                     shuffle=True,
                     partial_batching=False,
                     threads=1,
                     epochs=1,
                     max_batches=-1):

        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            actions = [env.action_space.sample() for env in self.envs]
        self.env.step_async(actions)
        obs, rewards, done, meta = self.env.step_wait()

        for i, val in enumerate([obs, actions, rewards, done, meta]):
            self.buffers[i].append(val)

        output = [np.array(buffer) for buffer in self.buffers]
        if self.done_policy(done):
            self.env.reset()
            self.buffer.clear()

        return output
