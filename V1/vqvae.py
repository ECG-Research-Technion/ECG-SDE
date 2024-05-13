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

class PositionalEncoding(nn.Module):

    def __init__(self, channel: int, dropout: float = 0.1, max_len: int = 1080):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channel, 2) * (-math.log(10000.0) / channel))
        pe = torch.zeros(max_len, 1, channel)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, channel]``
        """
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
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


class ResBlockEnc(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=channel*2,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                in_channels=channel*2,
                out_channels=in_channel,
                kernel_size=1,
            ),
            nn.SiLU(inplace=True),
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = out + inputs

        return out
    

class ResBlockDec(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channel,
                out_channels=channel*2,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(
                in_channels=channel*2,
                out_channels=in_channel,
                kernel_size=1,
            ),
            nn.SiLU(inplace=True),
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = out + inputs

        return out


class Encoder(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
            n_res_block: int,
            n_res_channel: int,
            additional_layers:int = 2,
            has_transformer: bool = False,
            has_attention: bool = False,
            n_trans_layers: int = 2,
    ):
        super().__init__()

        self.feature_expander = nn.Sequential(*[
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=channel,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(inplace=True),
        ])

        self.has_transformer = has_transformer
        if self.has_transformer:
            self.revin_layer = RevIN(num_features=channel, affine=True)
            self.transformer = Transformer(num_heads=2, num_layers=n_trans_layers, feature_dim=channel, sequence_len=forecast_signals, dropout=0.1)
        
        self.has_attention = has_attention
        if self.has_attention:
            self.pos_encoding = PositionalEncoding(channel=channel)
            self.attention = nn.MultiheadAttention(embed_dim=channel, num_heads=2, batch_first=True)
            self.fc = nn.Linear(channel*1080, channel*1080)
            self.fc_activation = nn.SiLU(inplace=True)
            self.normLayer1 = nn.BatchNorm1d(channel)
            self.normLayer2 = nn.BatchNorm1d(channel)

        blocks = []
        for _ in range(additional_layers):
            blocks.extend([
                nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel*2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SiLU(inplace=True),
                nn.Conv1d(
                    in_channels=channel*2,
                    out_channels=channel*2,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                ),
                nn.SiLU(inplace=True),
                nn.Conv1d(
                    in_channels=channel*2,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm1d(channel),
                nn.SiLU(inplace=True)
            ])

        for _ in range(n_res_block):
            blocks.append(ResBlockEnc(channel, n_res_channel))
            blocks.append(nn.BatchNorm1d(channel))

        blocks.append(nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                ))
                            
        self.blocks = nn.Sequential(*blocks)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs: Tensor):
        assert not (self.has_attention and self.has_transformer)
        out = self.feature_expander(inputs)
        if self.has_transformer:
            out = out.permute(0, 2, 1).contiguous()
            out = self.revin_layer(out, mode="norm")
            out = self.transformer(out)
            out = self.revin_layer(out, mode="denorm")
            out = out.permute(0, 2, 1).contiguous()

        if self.has_attention:
            out = out.permute(0, 2, 1).contiguous()
            out = self.pos_encoding(out)
            attn_out, _ = self.attention(out, out, out)
            out = out + attn_out
            out = out.permute(0, 2, 1).contiguous()
            out = self.normLayer1(out)
            orig_shape = out.shape
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = out.view(orig_shape)
            out = self.fc_activation(out)
            out = self.normLayer2(out)
            assert out.shape == orig_shape
        out = self.blocks(out)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            out_channel: int,
            channel: int,
            n_res_block: int,
            n_res_channel: int,
            additional_layers:int = 2,
            has_transformer: bool = False,
            has_attention: bool = False,
            n_trans_layers: int = 2,
    ):
        super().__init__()
        
        blocks = []

        blocks.append(nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                ))

        for _ in range(n_res_block):
            blocks.append(ResBlockDec(channel, n_res_channel))
            blocks.append(nn.BatchNorm1d(channel))
        
        for _ in range(additional_layers):
            blocks.extend(
                [
                    nn.ConvTranspose1d(
                        in_channels=channel,
                        out_channels=channel*2,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(inplace=True),
                    nn.ConvTranspose1d(
                        in_channels=channel*2,
                        out_channels=channel*2,
                        kernel_size=4,
                        padding=1,
                        stride=2,
                    ),
                    nn.SiLU(inplace=True),
                    nn.ConvTranspose1d(
                        in_channels=channel*2,
                        out_channels=channel,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(channel),
                    nn.SiLU(inplace=True),
                ]
            )

        self.has_transformer = has_transformer
        if self.has_transformer:
            self.revin_layer = RevIN(num_features=channel, affine=True)
            self.transformer = Transformer(num_heads=2, num_layers=n_trans_layers, feature_dim=channel, sequence_len=forecast_signals, dropout=0.1)

        self.has_attention = has_attention
        if self.has_attention:
            self.pos_encoding = PositionalEncoding(channel=channel)
            self.attention = nn.MultiheadAttention(embed_dim=channel, num_heads=2, batch_first=True)
            self.fc = nn.Linear(channel*1080, channel*1080)
            self.fc_activation = nn.SiLU(inplace=True)
            self.normLayer1 = nn.BatchNorm1d(channel)
            self.normLayer2 = nn.BatchNorm1d(channel)

        self.feature_reducer = nn.Sequential(*[
            nn.ConvTranspose1d(
                in_channels=channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            )
        ])

        self.blocks = nn.Sequential(*blocks)

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        assert not (self.has_attention and self.has_transformer)
        out = self.blocks(inputs)
        if self.has_transformer:
            out = out.permute(0, 2, 1).contiguous()
            out = self.revin_layer(out, mode="norm")
            out = self.transformer(out)
            out = self.revin_layer(out, mode="denorm")
            out = out.permute(0, 2, 1).contiguous()

        if self.has_attention:
            out = out.permute(0, 2, 1).contiguous()
            out = self.pos_encoding(out)
            attn_out, _ = self.attention(out, out, out)
            out = out + attn_out
            out = out.permute(0, 2, 1).contiguous()
            out = self.normLayer1(out)
            orig_shape = out.shape
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = out.view(orig_shape)
            out = self.fc_activation(out)
            out = self.normLayer2(out)
            assert out.shape == orig_shape

        out = self.feature_reducer(out)
        return out


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel: int = 2,
            channel: int = 64,
            n_res_block: int = 15,
            n_res_channel: int = 64,
            embed_dim: int = 512,
            n_embed: int = 512,
            decay: float = 0.99,
            n_dims: int = 1,
            has_transformer: bool = False,
            has_attention: bool = False,
            n_trans_layers: int = 2,
            additional_layers: int = 2,
    ):
        super().__init__()

        self._n_dims = n_dims
        self.enc_b = Encoder(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            additional_layers=additional_layers,
            has_transformer=has_transformer,
            has_attention=has_attention,
            n_trans_layers=n_trans_layers,
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
            out_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            additional_layers=additional_layers,
            has_transformer=has_transformer,
            has_attention=has_attention,
            n_trans_layers=n_trans_layers,
        )

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
        enc_b = self.enc_b(inputs)
        
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1)
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


# class VQVAEWrapper(nn.Module):
#     """
#     A class which wraps a pre-trained VQ-VAE model around another generative model, M, which then operates in the VQ-VAE
#     latent space, i.,e. the complete model performs x-> Encoder -> M -> Decoder -> y.
#     """

#     def __init__(
#             self,
#             pre_trained_vqvae_path: str,
#             pre_trained_vqvae_params: Dict[str, Any],
#             generative_model_top: nn.Module,
#             generative_model_middle: nn.Module,
#             generative_model_bottom: nn.Module,
#             device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ):
#         super().__init__()

#         ckpt = torch.load(pre_trained_vqvae_path, map_location=device)['model']
#         vqvae = VQVAE(
#             **pre_trained_vqvae_params
#         )
#         vqvae.load_state_dict(ckpt, strict=True)
#         vqvae = vqvae.to(device)
#         vqvae.requires_grad_(False)

#         self.vqvae = vqvae
#         self.generative_model_bottom = generative_model_bottom

#     def __call__(self, x: Tensor) -> Dict:
#         return self.forward(x)

#     def forward(self, x: Tensor) -> Dict:
#         # Encode the inputs through the VQ-VAE encoder
#         quant_b, diff, _ = self.vqvae.encode(x)

#         # Apply the generative model to the latent representations
#         pred_b = self.generative_model_bottom(quant_b)

#         # Decode only the next prediction
#         p_b = pred_b[MODELS_TENSOR_PREDICITONS_KEY]
#         prediction_horizon = p_t.shape[2] // quant_t.shape[2]
#         if prediction_horizon > 1:
#             p_t = p_t[..., :quant_t.shape[2], :]
#             p_m = p_m[..., :quant_m.shape[2], :]
#             p_b = p_b[..., :quant_b.shape[2], :]

#         # Apply the decoder to the outputs of the generative model
#         pred = self.vqvae.decode(
#             quant_t=p_t,
#             quant_m=p_m,
#             quant_b=p_b,
#         )

#         out = {
#             MODELS_TENSOR_PREDICITONS_KEY: pred,
#             OTHER_KEY: {
#                 'latent_loss': diff,
#                 'pred_t': pred_t[MODELS_TENSOR_PREDICITONS_KEY],
#                 'pred_m': pred_m[MODELS_TENSOR_PREDICITONS_KEY],
#                 'pred_b': pred_b[MODELS_TENSOR_PREDICITONS_KEY],
#                 'quant_t': quant_t,
#                 'quant_m': quant_m,
#                 'quant_b': quant_b,
#             }
#         }

#         return out

    
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