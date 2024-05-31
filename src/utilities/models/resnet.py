import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self, num_classes: int, resnet: str = 'resnet50'):
        super().__init__()

        # Load pretrained ResNet model
        if resnet == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif resnet == 'resnet34':
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif resnet == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f'Invalid resnet type: {resnet}')
            
        # Remove the original fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Forward pass through ResNet model
        x = self.model(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


class ResNetWithMetadata(nn.Module):
    def __init__(self, num_classes: int, resnet: str = 'resnet50'):
        super().__init__()

        # Load pretrained ResNet model
        if resnet == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif resnet == 'resnet34':
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif resnet == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f'Invalid resnet type: {resnet}')
        
        # Remove the original fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 64)
        self.metadata_fc_layer = nn.Linear(16, 64)
        self.middle_fc_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_classes)

    
    def forward(self, images, meta_features):
        # Forward pass through ResNet model
        x = self.model(images)
        y = F.leaky_relu(self.metadata_fc_layer(meta_features))
        x = torch.cat((x, y), dim=1)

        x = F.leaky_relu(self.middle_fc_layer(x))
        x = self.output_layer(x)
        
        # Apply softmax activation to the output
        x = F.softmax(x, dim=1)
        return x


def get_resnet(num_classes: int, use_metadata: bool, resnet: str = 'resnet50'):
    if use_metadata:
        return ResNetWithMetadata(num_classes, resnet)
    return ResNet(num_classes, resnet)
