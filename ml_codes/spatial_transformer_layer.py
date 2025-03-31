import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt


class SpatialTransformer(nn.Module):
    def __init__(self, inchannels=64, outchannels=64):
        super().__init__()
        self.inchannels = inchannels
        self.affine_parameters = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, (3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inchannels, inchannels, (3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inchannels, inchannels, (3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(inchannels * 4 * 4, 6)

    def forward(self, x):
        theta = self.fc(
            self.affine_parameters(x).view(-1, self.inchannels * 4 * 4)
        ).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        sampled = F.grid_sample(x, grid)
        return sampled


def test_spatial_transformer_with_image():

    # Read and preprocess image
    img = cv2.imread("512-grayscale-image-Cameraman.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))

    # Convert to tensor and add batch and channel dimensions (B,C,H,W)
    x = torch.from_numpy(img).float()
    x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Normalize to [-1, 1] range
    x = (x / 255.0) * 2 - 1

    # Create model and move everything to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialTransformer(inchannels=1, outchannels=1).to(device)
    x = x.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Convert tensors back to numpy for visualization
    input_img = x.cpu().squeeze().numpy()
    output_img = output.cpu().squeeze().numpy()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(input_img, cmap="gray")
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(output_img, cmap="gray")
    ax2.set_title("Transformed Image")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("spatial_transformer_result.png")
    plt.show()


if __name__ == "__main__":
    test_spatial_transformer_with_image()
