import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from paths import *


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
    ):
        super().__init__()

        self.LeakyReLU = nn.SiLU()
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

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        out = self.conv(inputs) + inputs
        out = self.silu(out)

        return out
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        latent_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return latent_loss, quantized


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        self.enc_b = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=embedding_dim, kernel_size=1),
            # nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
        )
        
        self.quantize_conv_b = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1)
        self.quantize_b = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(in_channels=embedding_dim, out_channels=256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=2, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, inputs):
        enc_b = self.enc_b(inputs)
        quant_b = self.quantize_conv_b(enc_b)
        diff, quant_b = self.quantize_b(quant_b)
        return quant_b, diff
    
    def decode(self, quant_b):
        dec = self.dec(quant_b)
        return dec

# Define the directory to save plots
plot_dir = 'vqvae_plots'
os.makedirs(plot_dir, exist_ok=True)

# Load tensor x from the specified path
x = torch.load('train_MITBIH_1_v001.pt')

# Define the VQVAE model
vqvae = VQVAE()
vqvae = vqvae.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vqvae.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000)

# Training parameters
num_epochs = 5000

# Move tensor to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)

# Training loop to overfit the tensor x
for epoch in range(num_epochs):
    vqvae.train()
    optimizer.zero_grad()
    
    # Forward pass
    quant_b, diff = vqvae.encode(x)
    x_reconstructed = vqvae.decode(quant_b)
    
    # Compute loss
    loss = criterion(x_reconstructed, x) + diff.mean()
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step(loss, epoch=epoch)
    
    # Print loss for debugging
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        
        for i in range(4):
            # Plot the reconstructed signal for the i-th element
            original_signal_1 = x[i, 0, :].detach().cpu().numpy()
            reconstructed_signal_1 = x_reconstructed[i, 0, :].detach().cpu().numpy()
            original_signal_2 = x[i, 1, :].detach().cpu().numpy()
            reconstructed_signal_2 = x_reconstructed[i, 1, :].detach().cpu().numpy()
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(original_signal_1, label='Original Signal - Dimension 1', color='orange')
            plt.plot(reconstructed_signal_1, label='Reconstructed Signal - Dimension 1', color='grey')
            plt.legend()
            plt.title(f'Epoch [{epoch + 1}/{num_epochs}], Signal {i+1} - Dimension 1')
            
            plt.subplot(2, 1, 2)
            plt.plot(original_signal_2, label='Original Signal - Dimension 2', color='orange')
            plt.plot(reconstructed_signal_2, label='Reconstructed Signal - Dimension 2', color='grey')
            plt.legend()
            plt.title(f'Epoch [{epoch + 1}/{num_epochs}], Signal {i+1} - Dimension 2')
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'epoch_{epoch + 1}_signal_{i+1}.png')
            plt.savefig(plot_path)
            plt.close()

# Evaluation
vqvae.eval()
with torch.no_grad():
    quant_b, _ = vqvae.encode(x)
    x_reconstructed = vqvae.decode(quant_b)

print("Original Tensor:")
print(x)
print("Reconstructed Tensor:")
print(x_reconstructed)