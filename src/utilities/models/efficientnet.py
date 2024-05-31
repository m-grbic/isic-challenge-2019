import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'efficientnet-b0'):
        super().__init__()

        # Load pretrained EfficientNet model
        self.model = EfficientNet.from_pretrained(model_name)

        # Remove the original fully connected layer
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Forward pass through EfficientNet model
        x = self.model(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


class EfficientNetWithMetadata(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'efficientnet-b0'):
        super().__init__()

        # Load pretrained EfficientNet model
        self.model = EfficientNet.from_pretrained(model_name)

        # Remove the original fully connected layer
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 64)
        self.metadata_fc_layer = nn.Linear(16, 64)
        self.middle_fc_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, images, meta_features):
        # Forward pass through EfficientNet model
        x = self.model(images)
        y = F.leaky_relu(self.metadata_fc_layer(meta_features))
        x = torch.cat((x, y), dim=1)

        x = F.leaky_relu(self.middle_fc_layer(x))
        x = self.output_layer(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


def get_efficientnet(num_classes: int, use_metadata: bool):
    if use_metadata:
        return EfficientNetWithMetadata(num_classes=num_classes)
    return EfficientNetModel(num_classes=num_classes)
