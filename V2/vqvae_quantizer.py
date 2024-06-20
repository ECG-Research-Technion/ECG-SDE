import torch
from torch import nn, Tensor


class Quantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, commitment_cost=0.25):
        super(Quantizer, self).__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.n_embed, self.embed_dim)
        self.embeddings.weight.data.uniform_(-1/self.n_embed, 1/self.n_embed)

    def forward(self, inputs: Tensor):
        # Flatten input
        flat_input = inputs.view(-1, self.embed_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embed, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        latent_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, latent_loss