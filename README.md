

# Setup

Library `timm` is used for ViT and ConvNeXt models, and the `efficientnet_pytorch` library for EfficientNet. Ensure you have these libraries installed:

```
pip install efficientnet_pytorch timm
```


## Inference speed benchmark

Benchmark was done on NVIDIA GeForce RTX 4070 Super with batch size 128 and 10 workers.

| use_metadata\model | ConvNext | ViT | EfficientNet | ResNet50 | ResNet34 | 
|---|---|---|---|---|---|
| false | 403 | 456 | 632 | 640 | 893
| true  | 394 | 468 | 678 | 622 | 879
