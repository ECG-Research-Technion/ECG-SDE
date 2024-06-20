import torch
from torch import nn, Tensor


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels: int = 2,
            enc_channels: int = 64,
    ):
        super().__init__()
        
        self.blocks = nn.Sequential(
            nn.ConvTranspose1d(in_channels=enc_channels, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
        )

    def __call__(self, inputs: Tensor):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.blocks(inputs)
        return out