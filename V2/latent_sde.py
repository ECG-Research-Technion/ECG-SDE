# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import argparse
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim
from transformer import TransformerWithResNet

import torchsde

# w/ underscore -> numpy; w/o underscore -> torch.
device = 'cuda'
latent_dim = 32*(128*5)


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):

    def __init__(self, theta=1.0, sigma=0.5): # There was theta for the mu. delete
        super(LatentSDE, self).__init__(noise_type="diagonal")
        logvar = math.log(sigma ** 2 / (2.))

        # Prior drift.
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # p(y0).
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = TransformerWithResNet(input_dim=latent_dim+1, model_dim=latent_dim, num_heads=1, num_layers=1, num_res_blocks=1, latent_dim=latent_dim)

        # q(y0).
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full((y.shape[0], 1), fill_value=t).to(device)
            
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.net(torch.cat((t, y), dim=1))

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), latent_dim)

    def h(self, t, y):  # Prior drift.
        return self.theta * (-y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:latent_dim]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:latent_dim]
        g = self.g(t, y)
        g_logqp = torch.zeros((y.shape[0], 1)).to(device)
        return torch.cat([g, g_logqp], dim=1)

    def forward(self, ts, y, batch_size, eps=None):
        # eps = torch.randn(batch_size, latent_dim).to(self.qy0_std) if eps is None else eps
        y0 = y # + eps * self.qy0_std
        qy0 = distributions.Normal(loc=y, scale=self.qy0_std)
        py0 = distributions.Normal(loc=y, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).

        aug_y0 = torch.cat([y0, torch.zeros(y0.size(0), 1).to(y0)], dim=1)
        aug_ys = torchsde.sdeint_adjoint(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method='euler',
            dt=0.01,
            adaptive=True,
            rtol=1e-3,
            atol=1e-3,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        
        ys, logqp_path = aug_ys[:, :, 0:latent_dim], aug_ys[-1, :, -1]
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

    def sample_p(self, ts, y, batch_size, eps=None, bm=None):
        # eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = y # + eps * self.py0_std
        return torchsde.sdeint_adjoint(self, y0, ts, bm=bm, method='srk', dt=1e-2, names={'drift': 'h'})

    def sample_q(self, ts, y, batch_size, eps=None, bm=None):
        # eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = y # + eps * self.qy0_std
        return torchsde.sdeint_adjoint(self, y0, ts, bm=bm, method='srk', dt=1e-2)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)
