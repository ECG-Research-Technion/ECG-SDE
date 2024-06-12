# Taken from https://github.com/rosinality/vq-vae-2-pytorch and modified to a 1D TCN variant,
# and to match the project API.

from typing import Dict, Any
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformer import *
from paths import *

MODELS_TENSOR_PREDICITONS_KEY = "pred_key"
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


class Quantize(nn.Module):
    def __init__(
            self,
            dim: int,
            n_embed: int,
            decay: float = 0.99,
            eps: float = 1e-5,
    ):
        super().__init__()

        self.dim = int(dim)
        self.n_embed = int(n_embed)
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.dim, self.n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs: Tensor):
        flatten = inputs.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        return quantize, diff, embed_ind

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def embed_code(self, embed_id: Tensor):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


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


class Encoder(nn.Module):
    def __init__(
            self,
            n_res_block: int,
            n_res_channel: int,
            in_channel: int = 2,
            enc_channels: int = 16,
            additional_layers:int = 2,
    ):
        super().__init__()

        blocks = []
        cur_channel = in_channel
        for _ in range(3):
            blocks.extend([
                nn.Conv1d(
                    in_channels=cur_channel,
                    out_channels=cur_channel*2,
                    kernel_size=1,
                ),
                nn.BatchNorm1d(cur_channel*2),
                nn.SiLU(inplace=True),
            ])
            cur_channel *= 2
            
        for _ in range(additional_layers):
            blocks.extend([
                nn.Conv1d(
                    in_channels=cur_channel,
                    out_channels=cur_channel,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                ),
                nn.BatchNorm1d(cur_channel),
                nn.SiLU(inplace=True),
            ])

        for _ in range(n_res_block):
            blocks.append(ResBlock(cur_channel, n_res_channel))

        blocks.append(nn.Conv1d(
                    in_channels=cur_channel,
                    out_channels=cur_channel,
                    kernel_size=1,
        ))
                            
        self.blocks = nn.Sequential(*blocks)
        self.enc_channels = enc_channels

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs: Tensor):
        out = self.blocks(inputs)
        assert out.shape[1] == self.enc_channels
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            channel: int,
            n_res_block: int,
            n_res_channel: int,
            additional_layers:int = 2,
    ):
        super().__init__()
        
        blocks = []
        blocks.extend([nn.Conv1d(
                        in_channels=channel,
                        out_channels=channel,
                        kernel_size=1,
                    ),
                    nn.SiLU(inplace=True)
        ])

        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        
        for _ in range(additional_layers):
            blocks.extend(
                [
                    nn.ConvTranspose1d(
                        in_channels=channel,
                        out_channels=channel,
                        kernel_size=4,
                        padding=1,
                        stride=2,
                    ),
                    nn.BatchNorm1d(channel),
                    nn.SiLU(inplace=True),
                ]
        )

        cur_channel = channel
        for _ in range(3):
            blocks.extend([
                nn.Conv1d(
                    in_channels=cur_channel,
                    out_channels=cur_channel // 2,
                    kernel_size=1,
                ),
                nn.BatchNorm1d(cur_channel // 2),
                nn.SiLU(inplace=True),
            ])
            cur_channel //= 2

        self.blocks = nn.Sequential(*blocks)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.blocks(inputs)
        return out


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


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel: int = 2,
            channel: int = 32,
            enc_channels: int = 16,
            n_res_block: int = 12,
            n_res_channel: int = 32,
            embed_dim: int = 512,
            n_embed: int = 512,
            decay: float = 0.99,
            n_trans_layers: int = 6,
            additional_layers: int = 3,
    ):
        super().__init__()

        self.enc_b = Encoder(
            in_channel=in_channel,
            enc_channels=enc_channels,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            additional_layers=additional_layers,
        )

        self.quantize_conv_b = nn.Conv1d(
            in_channels=channel,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.quantize_b = Quantize(
            dim=embed_dim,
            n_embed=n_embed,
            decay=decay,
        )
        
        self.fix_channels = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=channel,
            kernel_size=1,
        )
        
        self.dec = Decoder(
            channel=enc_channels,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            additional_layers=additional_layers,
        )

        self.attn = MultiHeadAttention(embed_dim=16, num_heads=2, ff_hidden_dim=16, num_layers=n_trans_layers, dropout=0.1)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Dict:
        quant_b, diff, _ = self.encode(inputs)
        dec = self.decode(quant_b)

        out = {
            MODELS_TENSOR_PREDICITONS_KEY: dec,
            OTHER_KEY: {
                'latent_loss': diff,
            }
        }

        return out

    def encode(self, inputs: Tensor):
        attn = enc_b = self.enc_b(inputs)
        if has_attention:
            enc_b = enc_b.permute(0,2,1)
            attn=self.attn(enc_b)
            attn = attn.permute(0,2,1)
        
        quant_b = self.quantize_conv_b(attn).permute(0, 2, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 2, 1)
        quant_b = self.fix_channels(quant_b)
        diff_b = diff_b.unsqueeze(0)
        
        return quant_b, diff_b, id_b

    def decode(self, quant_b: Tensor):
        dec = self.dec(quant_b)

        return dec

    def decode_code(self, code_b: Tensor):
        
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 2, 1)

        dec = self.decode(quant_b)

        return dec


class ModuleLoss:
    def __init__(self, model: nn.Module, scale: float = 1.0):
        self.model = model
        self.scale = scale

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self.model(y_pred, y_true)
        loss = loss * self.scale
        return loss

    
class VQVAE_Loss():
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 1.0,
            distance_metric: ModuleLoss = ModuleLoss(
                model=nn.MSELoss(),
                scale=1,
            ),
    ):

        self._alpha = alpha
        self._gamma = gamma
        self._distance_metric = distance_metric

    def forward(self, inputs: Dict) -> Tensor:
        latent_loss = inputs[OTHER_KEY]['latent_loss']
        loss = (self._gamma * self._distance_metric(inputs, inputs[MODELS_TENSOR_PREDICITONS_KEY])) + (self._alpha * latent_loss.mean())

        return loss