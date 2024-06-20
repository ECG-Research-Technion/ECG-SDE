from typing import Dict
import torch
from torch import nn, Tensor
from paths import *
from vqvae_encoder import Encoder
from vqvae_decoder import Decoder
from vqvae_quantizer import Quantizer

MODELS_TENSOR_PREDICITONS = "pred_key"
OTHER_KEY = "other_key"

forecast_signals = config['samples_per_second']*config['input_signal_duration']*config['forecast_predictions']
has_attention = config['has_attention']

class RevIN(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = (x.ndim-1,)

        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev
        x = x + self.mean

        return x

    def __call__(self, x, mode: str):
        return self.forward(x=x, mode=mode)
    
    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, ff_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_hidden_dim, embed_dim),
                    nn.LayerNorm(embed_dim)
                )
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for attn, norm, ff in self.layers:
            attn_output, _ = attn(x, x, x)
            x = norm(x + attn_output)
            ff_output = ff(x)
            x = ff_output + x
        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=channel*2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(channel*2),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                in_channels=channel*2,
                out_channels=in_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(in_channel),
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.conv(inputs) + inputs
        out = self.silu(out)

        return out


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channels: int = 2,
            enc_channels: int = 32,
            embed_dim: int = 64,
            n_embed: int = 512,
    ):
        super().__init__()

        self.encode = Encoder(
            in_channels=in_channels,
            enc_channels=enc_channels,
        )

        self.pre_quantize = nn.Conv1d(
            in_channels=enc_channels,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.quantize = Quantizer(
            embed_dim=embed_dim,
            n_embed=n_embed,
        )
        
        self.post_quantize = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=enc_channels,
            kernel_size=1,
        )
        
        self.decode = Decoder(
            enc_channels=enc_channels,
            out_channels=in_channels,
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Dict:
        enc = self.encode(inputs)
        pre_quantize = self.pre_quantize(enc)
        quantized, latent_loss = self.quantize(pre_quantize)
        post_quantize = self.post_quantize(quantized)
        dec = self.decode(post_quantize)

        out = {
            'reconstructed': dec,
            'latent_loss': latent_loss,
        }

        return out