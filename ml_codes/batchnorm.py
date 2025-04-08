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


class BatchNorm2d(nn.Module):
    def __init__(self, feat_dim, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()  # Call parent constructor
        self.feat_dim = feat_dim
        self.eps = eps
        self.momentum = momentum

        # Initialize parameters
        self.gamma = nn.Parameter(torch.ones(feat_dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(feat_dim, 1, 1))

        # Running stats for inference mode
        self.register_buffer("running_mean", torch.zeros(feat_dim, 1, 1))
        self.register_buffer("running_var", torch.ones(feat_dim, 1, 1))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mu
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var
        else:
            # Use running statistics for inference
            mu = self.running_mean
            var = self.running_var

        # Normalize
        x_normalized = (x - mu) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_normalized + self.beta

        return out


# Define a simple CNN with and without BatchNorm for comparison
class SimpleCNN(nn.Module):
    def __init__(self, use_batchnorm=True, custom_batchnorm=False):
        super(SimpleCNN, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.custom_batchnorm = custom_batchnorm

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Batch normalization after first conv
        if use_batchnorm:
            if custom_batchnorm:
                self.bn1 = BatchNorm2d(32, eps=1e-5)
            else:
                self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Batch normalization after second conv
        if use_batchnorm:
            if custom_batchnorm:
                self.bn2 = BatchNorm2d(64, eps=1e-5)
            else:
                self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second convolutional block
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten and pass through fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
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


# Testing function
def test(model, device, test_loader, criterion):
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
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    return test_loss, accuracy


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 64
    epochs = 5
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
    model_without_bn = SimpleCNN(use_batchnorm=False).to(device)
    model_with_bn = SimpleCNN(use_batchnorm=True, custom_batchnorm=False).to(device)
    model_with_custom_bn = SimpleCNN(use_batchnorm=True, custom_batchnorm=True).to(
        device
    )

    # Define optimizers
    optimizer_without_bn = optim.SGD(model_without_bn.parameters(), lr=lr, momentum=0.9)
    optimizer_with_bn = optim.SGD(model_with_bn.parameters(), lr=lr, momentum=0.9)
    optimizer_with_custom_bn = optim.SGD(
        model_with_custom_bn.parameters(), lr=lr, momentum=0.9
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Track metrics
    train_losses = {"without_bn": [], "with_bn": [], "with_custom_bn": []}
    train_accuracies = {"without_bn": [], "with_bn": [], "with_custom_bn": []}
    test_losses = {"without_bn": [], "with_bn": [], "with_custom_bn": []}
    test_accuracies = {"without_bn": [], "with_bn": [], "with_custom_bn": []}

    # Train and evaluate each model
    models = [
        ("without_bn", model_without_bn, optimizer_without_bn),
        ("with_bn", model_with_bn, optimizer_with_bn),
        ("with_custom_bn", model_with_custom_bn, optimizer_with_custom_bn),
    ]

    for name, model, optimizer in models:
        print(f"\nTraining model: {name}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, device, train_loader, optimizer, criterion, epoch
            )
            test_loss, test_acc = test(model, device, test_loader, criterion)

            train_losses[name].append(train_loss)
            train_accuracies[name].append(train_acc)
            test_losses[name].append(test_loss)
            test_accuracies[name].append(test_acc)

        end_time = time.time()
        print(f"Training time for {name}: {end_time - start_time:.2f} seconds")

    # Plot results
    plt.figure(figsize=(14, 10))

    # Plot training loss
    plt.subplot(2, 2, 1)
    for name in train_losses.keys():
        plt.plot(train_losses[name], label=name)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot training accuracy
    plt.subplot(2, 2, 2)
    for name in train_accuracies.keys():
        plt.plot(train_accuracies[name], label=name)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot test loss
    plt.subplot(2, 2, 3)
    for name in test_losses.keys():
        plt.plot(test_losses[name], label=name)
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot test accuracy
    plt.subplot(2, 2, 4)
    for name in test_accuracies.keys():
        plt.plot(test_accuracies[name], label=name)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("batchnorm_comparison.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'batchnorm_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()
