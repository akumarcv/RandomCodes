import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


class LayerNorm2d(nn.Module):
    def __init__(self, feat_dim, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.feat_dim = feat_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, feat_dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))

    def forward(self, x):

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_normalized + self.beta


class UnitTests:
    @staticmethod
    def test_layernorm_output_shape():
        # Create random input tensor
        batch_size = 32
        channels = 16
        height = 24
        width = 24
        x = torch.randn(batch_size, channels, height, width)

        # Initialize layer norm
        layer_norm = LayerNorm2d(channels)

        # Forward pass
        output = layer_norm(x)

        # Check output shape matches input shape
        assert (
            output.shape == x.shape
        ), f"Output shape {output.shape} doesn't match input shape {x.shape}"
        print("✓ LayerNorm output shape test passed")

        return True

    @staticmethod
    def test_layernorm_normalization():
        # Create simple input tensor with known statistics
        batch_size = 2
        channels = 3
        height = 4
        width = 4
        x = torch.ones(batch_size, channels, height, width)

        # Add different values to create non-zero mean and variance
        x[0, 0] = 2.0  # Set first channel of first sample to 2.0
        x[1, 1] = 0.0  # Set second channel of second sample to 0.0

        # Initialize layer norm
        layer_norm = LayerNorm2d(channels)

        # Forward pass
        output = layer_norm(x)

        # Check mean and std for each sample in the batch
        for i in range(batch_size):
            sample_mean = output[i].mean().item()
            sample_std = output[i].std(unbiased=False).item()

            # Test if mean is close to 0 and std is close to 1
            assert (
                abs(sample_mean) < 1e-5
            ), f"Mean of sample {i} is {sample_mean}, expected close to 0"
            assert (
                abs(sample_std - 1.0) < 1e-4
            ), f"Std of sample {i} is {sample_std}, expected close to 1"

        print("✓ LayerNorm normalization test passed")
        return True

    @staticmethod
    def test_against_pytorch_implementation():
        # Create random input tensor
        batch_size = 16
        channels = 32
        height = 8
        width = 8
        x = torch.randn(batch_size, channels, height, width)

        # Initialize custom layer norm
        custom_layer_norm = LayerNorm2d(channels)

        # Initialize PyTorch's LayerNorm
        # Note: PyTorch's LayerNorm normalizes over the last dimensions,
        # so we need to permute our input and output to match
        pytorch_layer_norm = nn.LayerNorm([channels, height, width])

        # Forward pass with custom implementation
        custom_output = custom_layer_norm(x)

        # Forward pass with PyTorch implementation
        pytorch_output = pytorch_layer_norm(x)

        # Check if outputs are similar
        diff = torch.abs(custom_output - pytorch_output).max().item()
        assert (
            diff < 1e-5
        ), f"Max difference between implementations is {diff}, expected less than 1e-5"

        print("✓ Comparison with PyTorch implementation passed")
        return True


class SimpleConvNet(nn.Module):
    def __init__(self, use_layer_norm=True, use_custom_layer_norm=True):
        super(SimpleConvNet, self).__init__()

        self.use_layer_norm = use_layer_norm
        self.use_custom_layer_norm = use_custom_layer_norm

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)

        # Layer normalization after first conv
        if use_layer_norm:
            if use_custom_layer_norm:
                self.ln1 = LayerNorm2d(16)
            else:
                self.ln1 = nn.LayerNorm([16, 28, 28])

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Layer normalization after second conv
        if use_layer_norm:
            if use_custom_layer_norm:
                self.ln2 = LayerNorm2d(32)
            else:
                self.ln2 = nn.LayerNorm([32, 14, 14])

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second convolutional block
        x = self.conv2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten and pass through fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f} Accuracy: {100 * correct / total:.2f}%"
            )

    return running_loss / len(train_loader), 100 * correct / total


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
    )

    return test_loss, accuracy


def compare_models():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 64
    epochs = 4
    lr = 0.01

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create models
    model_without_ln = SimpleConvNet(use_layer_norm=False).to(device)
    model_with_torch_ln = SimpleConvNet(
        use_layer_norm=True, use_custom_layer_norm=False
    ).to(device)
    model_with_custom_ln = SimpleConvNet(
        use_layer_norm=True, use_custom_layer_norm=True
    ).to(device)

    # Create optimizers
    optimizer_without_ln = optim.Adam(model_without_ln.parameters(), lr=lr)
    optimizer_with_torch_ln = optim.Adam(model_with_torch_ln.parameters(), lr=lr)
    optimizer_with_custom_ln = optim.Adam(model_with_custom_ln.parameters(), lr=lr)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    models = [
        {
            "name": "Without LayerNorm",
            "model": model_without_ln,
            "optimizer": optimizer_without_ln,
        },
        {
            "name": "With PyTorch LayerNorm",
            "model": model_with_torch_ln,
            "optimizer": optimizer_with_torch_ln,
        },
        {
            "name": "With Custom LayerNorm",
            "model": model_with_custom_ln,
            "optimizer": optimizer_with_custom_ln,
        },
    ]

    results = {
        "train_loss": {model["name"]: [] for model in models},
        "train_acc": {model["name"]: [] for model in models},
        "test_loss": {model["name"]: [] for model in models},
        "test_acc": {model["name"]: [] for model in models},
        "time": {model["name"]: 0 for model in models},
    }

    for model_info in models:
        model_name = model_info["name"]
        model = model_info["model"]
        optimizer = model_info["optimizer"]

        print(f"\n=== Training {model_name} ===")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch
            )
            test_loss, test_acc = evaluate(model, device, test_loader, criterion)

            results["train_loss"][model_name].append(train_loss)
            results["train_acc"][model_name].append(train_acc)
            results["test_loss"][model_name].append(test_loss)
            results["test_acc"][model_name].append(test_acc)

        end_time = time.time()
        training_time = end_time - start_time
        results["time"][model_name] = training_time
        print(f"Training time for {model_name}: {training_time:.2f} seconds")

    # Plot results
    plt.figure(figsize=(16, 12))

    # Plot training loss
    plt.subplot(2, 2, 1)
    for model_name in results["train_loss"]:
        plt.plot(results["train_loss"][model_name], marker="o", label=model_name)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot test loss
    plt.subplot(2, 2, 2)
    for model_name in results["test_loss"]:
        plt.plot(results["test_loss"][model_name], marker="o", label=model_name)
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for model_name in results["train_acc"]:
        plt.plot(results["train_acc"][model_name], marker="o", label=model_name)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Plot test accuracy
    plt.subplot(2, 2, 4)
    for model_name in results["test_acc"]:
        plt.plot(results["test_acc"][model_name], marker="o", label=model_name)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("layernorm_comparison.png", dpi=300)
    plt.show()

    # Print training times
    print("\nTraining Times:")
    for model_name, train_time in results["time"].items():
        print(f"{model_name}: {train_time:.2f} seconds")

    # Print final accuracies
    print("\nFinal Test Accuracies:")
    for model_name in results["test_acc"]:
        print(f"{model_name}: {results['test_acc'][model_name][-1]:.2f}%")


def main():
    print("Running unit tests for LayerNorm2d...")
    try:
        UnitTests.test_layernorm_output_shape()
        UnitTests.test_layernorm_normalization()
        UnitTests.test_against_pytorch_implementation()
        print("\nAll unit tests passed!\n")
    except AssertionError as e:
        print(f"Test failed: {e}")
        return

    print("\nComparing models with and without LayerNorm...")
    compare_models()


if __name__ == "__main__":
    main()
