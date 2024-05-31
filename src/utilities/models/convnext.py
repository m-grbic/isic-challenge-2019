import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'convnext_base'):
        super().__init__()

        # Load pretrained ConvNeXt model
        self.model = timm.create_model(model_name, pretrained=True)

        # Remove the original fully connected layer
        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Forward pass through ConvNeXt model
        x = self.model(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


class ConvNeXtWithMetadata(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'convnext_base'):
        super().__init__()

        # Load pretrained ConvNeXt model
        self.model = timm.create_model(model_name, pretrained=True)

        # Remove the original fully connected layer
        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(num_features, 64)
        self.metadata_fc_layer = nn.Linear(16, 64)
        self.middle_fc_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, images, meta_features):
        # Forward pass through ConvNeXt model
        x = self.model(images)
        y = F.leaky_relu(self.metadata_fc_layer(meta_features))
        x = torch.cat((x, y), dim=1)

        x = F.leaky_relu(self.middle_fc_layer(x))
        x = self.output_layer(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


def get_convnext(num_classes: int, use_metadata: bool):
    if use_metadata:
        return ConvNeXtWithMetadata(num_classes=num_classes)
    return ConvNeXtModel(num_classes=num_classes)
