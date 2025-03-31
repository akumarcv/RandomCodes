import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3  # pip install timm
from tqdm import tqdm  # pip install tqdm
import matplotlib.pyplot as plt  # pip install matplotlib
import torch.optim as optim
import numpy as np
import pdb


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]


def test_sinusoidal_embedding():
    """
    Test the SinusoidalEmbedding class with visualization
    """
    # Test parameters
    time_steps = 100
    embedding_dim = 32
    batch_size = 1

    # Initialize the embedding layer
    embedding = SinusoidalEmbeddings(time_steps, embedding_dim)

    # Create dummy input tensor (B, C, H, W)
    x = torch.randn(batch_size, 3, 32, 32)

    # Generate positions for testing
    t = torch.arange(time_steps)

    # Get embeddings
    with torch.no_grad():
        embeddings = embedding(x, t)

    # Squeeze extra dimensions for plotting
    embeddings = embeddings.squeeze(-1).squeeze(-1)

    # Convert to numpy for plotting
    embeddings_np = embeddings.numpy()
    t_np = t.numpy()

    # Visualize the embeddings
    plt.figure(figsize=(12, 6))

    # Plot first few dimensions
    num_dims_to_plot = 4
    for i in range(num_dims_to_plot):
        plt.plot(t_np, embeddings_np[:, i], label=f"Dimension {i}")

    plt.title("Sinusoidal Time Step Embeddings")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("sinusoidal_embeddings.png")
    plt.show()

    # Print embedding information
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Time steps: {time_steps}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    return embeddings


if __name__ == "__main__":
    embeddings = test_sinusoidal_embedding()
