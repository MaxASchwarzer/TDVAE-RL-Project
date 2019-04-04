from pylego.model import Model
from abc import ABC, abstractmethod
import contextlib
import glob
import pathlib

import torch
from torch import autograd, nn, optim
import torchvision


class BaseGymTDVAE(Model):

    def __init__(self, model, flags, *args, **kwargs):
        self.flags = flags
        super().__init__(model=model, *args, **kwargs)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor()
        ])

    def prepare_batch(self, data):
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]
        if self.is_training():
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()
        with context:
            data = tuple(torch.as_tensor(d, device=self.device) for d in data)
        if len(data) == 1:
            data = data[0]
        return data
