import torch
from torch import nn, Tensor


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32, out_channels: int = 2):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(in_channels=64, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs: Tensor):
        x = self.blocks(inputs)
        # x = inputs
        # for layer in self.blocks:
        #     if isinstance(layer, nn.LayerNorm):
        #         x = x.permute(0, 2, 1)
        #         x = layer(x)
        #         x = x.permute(0, 2, 1)
        #     else:
        #         x = layer(x)

        return x