import torch
from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 2,
            latent_dim: int = 32,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=latent_dim, kernel_size=3, padding=1),
            nn.Tanh(),
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