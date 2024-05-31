import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim=64, sequence_len=640):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(sequence_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * -(math.log(10000.0) / feature_dim))
        pe = torch.zeros(sequence_len, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [batch_size, seq_length, feature_dim]
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        # x is expected to be of shape [batch_size, seq_length, feature_dim]
        x = x + self.positional_encoding
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads=4, feature_dim=64, sequence_len=640, dropout=0.1):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(feature_dim=feature_dim, sequence_len=sequence_len)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        return x

# import math
# import torch
# from torch import nn
# from flash_attn.models.gpt import GPTLMHeadModel
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# class PositionalEncoding(nn.Module):
#     def __init__(self, feature_dim=2, sequence_len=640):
#         super(PositionalEncoding, self).__init__()

#         position = torch.arange(sequence_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, feature_dim, 2) * -(math.log(10000.0) / feature_dim))
#         pe = torch.zeros(sequence_len, feature_dim)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         pe = pe.unsqueeze(0)  # Shape: [batch_size, seq_length, feature_dim]
#         self.register_buffer('positional_encoding', pe)

#     def forward(self, x):
#         # x is expected to be of shape [batch_size, seq_length, feature_dim]
#         x = x + self.positional_encoding
#         return x


# class TransformerWithFlashAttention(nn.Module):
#     def __init__(self, num_layers, num_heads=1, feature_dim=2, sequence_len=640, dropout=0.1):
#         super(TransformerWithFlashAttention, self).__init__()
#         self.positional_encoding = PositionalEncoding(feature_dim=feature_dim, sequence_len=sequence_len)
#         self.dropout = nn.Dropout(p=dropout)

#         # Create the configuration for GPT model
#         config = GPT2Config(
#             vocab_size=50257,
#             n_positions=sequence_len,
#             n_embd=feature_dim,
#             n_layer=num_layers,
#             n_head=num_heads,
#             scale_attn_by_inverse_layer_idx=True,
#             rotary_emb_fraction=0.5,
#             use_flash_attn=True,
#             fused_mlp=True,
#             fused_bias_fc=True,
#             fused_dropout_add_ln=True,
#             pad_vocab_size_multiple=8
#         )

#         # Initialize the GPT model with the specified configuration
#         self.transformer_encoder = GPTLMHeadModel(config)

#     def forward(self, x):
#         x = self.positional_encoding(x)
#         x = self.dropout(x)
#         x = self.transformer_encoder(x)
#         return x
