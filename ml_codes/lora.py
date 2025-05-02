import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleConvNet(nn.Module):
    """A simple 3-layer convolutional neural network."""
    
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleConvNet, self).__init__()
        
        # First convolutional layer: input_channels -> 16 channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, 
                              kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer: 16 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third convolutional layer: 32 -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                              kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layer (assuming input size is 32x32)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        
    def forward(self, x):
        # First layer: conv -> batch norm -> ReLU -> max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second layer: conv -> batch norm -> ReLU -> max pooling
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Third layer: conv -> batch norm -> ReLU -> max pooling
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten and feed to fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module.
    
    This implementation adds low-rank adaptations to a linear layer.
    """
    
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRA, self).__init__()
        
        # Scaling factor
        self.alpha = alpha
        self.rank = rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize with random values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Low-rank adaptation: x -> A -> B
        return (self.alpha / self.rank) * (x @ self.lora_A @ self.lora_B)


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LinearWithLoRA, self).__init__()
        
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # LoRA adaptation
        self.lora = LoRA(in_features, out_features, rank, alpha)
        
    def forward(self, x):
        # Original output + LoRA adaptation
        return self.linear(x) + self.lora(x)


class ConvLoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for convolutional layers.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, alpha=1.0):
        super(ConvLoRA, self).__init__()
        
        # Scaling factor
        self.alpha = alpha
        self.rank = rank
        
        # Low-rank decomposition for convolution
        # First map input channels to rank dimensions
        self.lora_A = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rank,
            kernel_size=1,  # Use 1x1 convolutions for dimension reduction
            bias=False
        )
        
        # Then map rank dimensions to output channels using the original kernel size
        self.lora_B = nn.Conv2d(
            in_channels=rank,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False
        )
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Apply low-rank adaptation: x -> A -> B
        return (self.alpha / self.rank) * self.lora_B(self.lora_A(x))


class ConvWithLoRA(nn.Module):
    """Convolutional layer with LoRA adaptation."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, rank=4, alpha=1.0):
        super(ConvWithLoRA, self).__init__()
        
        # Original convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # LoRA adaptation
        self.lora = ConvLoRA(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            rank=rank,
            alpha=alpha
        )
        
        # Save parameters for potential use
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
    def forward(self, x):
        # Original output + LoRA adaptation
        return self.conv(x) + self.lora(x)


# Updated conversion function to handle both linear and conv layers
def convert_model_to_lora(model, rank=4, alpha=1.0):
    """Convert a model's linear and convolutional layers to LoRA-adapted layers."""
    
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            # Create new LoRA layer with weights from original layer
            lora_layer = LinearWithLoRA(in_features, out_features, rank, alpha)
            lora_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.linear.bias.data = module.bias.data.clone()
            
            # Replace the original layer
            setattr(model, name, lora_layer)
            
        elif isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None
            
            # Create new LoRA layer with weights from original layer
            lora_layer = ConvWithLoRA(
                in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias, rank, alpha
            )
            lora_layer.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.conv.bias.data = module.bias.data.clone()
            
            # Replace the original layer
            setattr(model, name, lora_layer)
            
        else:
            # Recursively apply to submodules
            convert_model_to_lora(module, rank, alpha)
    
    return model


# Usage example:
if __name__ == "__main__":
    # Create a CNN model
    model = SimpleConvNet()
    
    # Convert to LoRA model for fine-tuning
    lora_model = convert_model_to_lora(model, rank=8)
    
    # In practice, you would:
    # 1. Train the original model first
    # 2. Freeze the model parameters
    # 3. Convert to LoRA and only train the LoRA parameters
    
    # Example of freezing original weights and only training LoRA weights
    for name, param in lora_model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True  # Train LoRA parameters
        else:
            param.requires_grad = False  # Freeze original parameters
    
    # Test forward pass with random data (assuming 32x32 RGB images)
    dummy_input = torch.randn(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
    output = lora_model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [4, 10] for 10 classes
    
    # Print model architecture to see the converted layers
    print("\nModel Architecture:")
    for name, module in lora_model.named_modules():
        if isinstance(module, (LinearWithLoRA, ConvWithLoRA)):
            print(f"{name}: {module.__class__.__name__}")
