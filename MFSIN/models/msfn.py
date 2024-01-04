import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers.drop import DropPath
import numpy as np

"""
Original TokenLearner Module modified from:
https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py
"""
class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim,  1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class MSFN(nn.Module):

    def __init__(self, in_dim, num_token, use_sum_pooling=False):
        super(MSFN, self).__init__()
        self.in_dim = in_dim
        self.num_token = num_token
        self.use_sum_pooling = use_sum_pooling

        self.selected_func = nn.Sequential(
            LayerNorm(self.in_dim),
            nn.Conv2d(self.in_dim, self.num_token, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_token, self.num_token, 3, 1, 1, bias=False),
            Rearrange('b n h w -> b n (h w)'),
            nn.Sigmoid()
        )

        self.feat_func = Rearrange('b c h w -> b (h w) c')

    def forward(self, inputs):
        # select func: layernorm -> 3 * (conv + gelu) -> conv -> reshape -> sigmoid
        # output shape: (batch_size, num_token, H * W, 1)

        selected = self.selected_func(inputs)
        selected = selected[:, :, :, None]

        # feat func: reshape
        # output shape: (batch_size, 1, H * W, channels)
        feat = self.feat_func(inputs)
        feat = feat[:, None, :, :]

        if self.use_sum_pooling:
            inputs = torch.sum(feat * selected, dim=2)
        else:
            inputs = torch.mean(feat * selected, dim=2)
        inputs = inputs.permute(0, 2, 1)

        return inputs


