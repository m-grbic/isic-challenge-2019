from .resnet import get_resnet
from .vit import get_vit
from .efficientnet import get_efficientnet
from .convnext import get_convnext


def get_model(model_name: str, num_classes: int, use_metadata: bool):

    if model_name.startswith("resnet"):
        print(f"Creating ResNet ({model_name})")
        return get_resnet(num_classes=num_classes, use_metadata=use_metadata, resnet=model_name)
    elif model_name.startswith("vit"):
        print("Creating ViT")
        return get_vit(num_classes=num_classes, use_metadata=use_metadata)
    elif model_name.startswith("convnext"):
        print("Creating ConvNext")
        return get_convnext(num_classes=num_classes, use_metadata=use_metadata)
    elif model_name.startswith("efficientnet"):
        print("Creating EfficientNet")
        return get_efficientnet(num_classes=num_classes, use_metadata=use_metadata)
