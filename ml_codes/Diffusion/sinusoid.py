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
import seaborn as sns 
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
    time_steps = 1000
    embedding_dim = 64
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

    # Create a figure with multiple plots
    plt.figure(figsize=(15, 10))
    
    # 1. Line plot of first few dimensions
    plt.subplot(2, 1, 1)
    num_dims_to_plot = 8
    for i in range(num_dims_to_plot):
        plt.plot(t_np, embeddings_np[:, i], label=f"Dimension {i}")

    plt.title("Sinusoidal Time Step Embeddings - Selected Dimensions")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True)
    
    # 2. Seaborn heatmap for 2D color map visualization
    plt.subplot(2, 1, 2)
    sns.heatmap(
        embeddings_np, 
        cmap="viridis",
        cbar_kws={"label": "Embedding Value"},
        xticklabels=10,  # Show fewer ticks for readability
        yticklabels=10
    )
    plt.title("Sinusoidal Embeddings 2D Color Map")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Time Step")
    
    plt.tight_layout()
    plt.savefig("sinusoidal_embeddings.png", dpi=300)
    plt.show()
    
    # Create a separate figure for a larger, more detailed heatmap
    plt.figure(figsize=(12, 8))
    
    # Use seaborn's custom color palette for more visual contrast
    # Diverging palette works well for values that go from negative to positive
    sns.heatmap(
        embeddings_np, 
        cmap="coolwarm",
        center=0,  # Center the colormap at zero
        robust=True,  # Use robust quantile-based color scaling
        cbar_kws={"label": "Embedding Value", "shrink": 0.8},
        xticklabels=8,
        yticklabels=10
    )
    
    plt.title("Detailed Sinusoidal Embedding Patterns")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Time Step (Diffusion Process)")
    
    # Add annotations to highlight the different frequencies
    plt.text(embedding_dim + 1, time_steps//2, 
             "Higher dimensions\nhave higher\nfrequencies", 
             fontsize=10, va="center")
    
    plt.tight_layout()
    plt.savefig("sinusoidal_embeddings_heatmap.png", dpi=300)
    plt.show()

    # Print embedding information
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Time steps: {time_steps}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    return embeddings


if __name__ == "__main__":
    embeddings = test_sinusoidal_embedding()