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
    """
    Sinusoidal position embeddings for diffusion timesteps.

    This class creates position embeddings using sine and cosine functions
    of different frequencies, following the approach used in the paper
    "Attention Is All You Need" and adapted for diffusion models.

    Parameters:
    -----------
    time_steps : int
        The total number of time steps in the diffusion process
    embed_dim : int
        The dimension of the embedding vector

    Attributes:
    -----------
    embeddings : torch.Tensor
        Pre-computed embedding vectors for each time step
    """

    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        # Create position tensor of shape (time_steps, 1)
        position = torch.arange(time_steps).unsqueeze(1).float()

        # Calculate division terms with exponentially decreasing frequencies
        # This creates different frequency bands for different dimensions
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        
        # Initialize empty embedding tensor
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)

        # Fill even indices with sin and odd indices with cos
        # This creates embeddings with different frequencies for different dimensions
        embeddings[:, 0::2] = torch.sin(position * div)  # Even dimensions
        embeddings[:, 1::2] = torch.cos(position * div)  # Odd dimensions

        self.embeddings = embeddings

    def forward(self, x, t):
        """
        Get the time embeddings for a batch of inputs at specified timesteps.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor, used only for device information
        t : torch.Tensor
            Time step indices

        Returns:
        --------
        torch.Tensor
            The time embeddings with shape (batch_size, embed_dim, 1, 1)
            suitable for adding to convolutional feature maps
        """
        # Select embeddings for the requested time steps and move to the correct device
        embeds = self.embeddings[t].to(x.device)

        # Add spatial dimensions for broadcasting with convolutional feature maps
        return embeds[:, :, None, None]


def test_sinusoidal_embedding():
    """
    Test the SinusoidalEmbedding class with visualization.

    This function generates sinusoidal embeddings for all time steps,
    and creates visualizations to help understand their properties:
    1. Line plots showing the embedding values for selected dimensions
    2. Heatmap showing the full embedding pattern across all dimensions and time steps
    3. A more detailed heatmap with annotations

    Returns:
    --------
    torch.Tensor
        The generated embeddings
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
        yticklabels=10,
    )

    plt.title("Detailed Sinusoidal Embedding Patterns")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Time Step (Diffusion Process)")

    # Add annotations to highlight the different frequencies
    plt.text(
        embedding_dim + 1,
        time_steps // 2,
        "Higher dimensions\nhave higher\nfrequencies",
        fontsize=10,
        va="center",
    )

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
