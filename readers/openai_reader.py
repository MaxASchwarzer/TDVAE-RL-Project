"""Some parts adapted from the TD-VAE code by Xinqiang Ding <xqding@umich.edu>."""

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
import gym
import gym-vecenv as vecenv

from pylego.reader import DatasetReader

class GymReader(Reader):

    def __init__(self,
                env,
                batch_size):

        self.envs = [gym.make(env) for i in range(batch_size)]
        self.env = vecenv.SubprocVecEnv(envs)

        super().__init__({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})


    def iter_batches(self,
                     split_name,
                     batch_size,
                     actions=None,
                     shuffle=True,
                     partial_batching=False,
                     threads=1,
                     epochs=1,
                     max_batches=-1):

        if actions=None:
            # We have to assume a random policy if nobody gives us actions
            actions = [env.action_space.sample() for env in self.envs]
