

# Setup

Library `timm` is used for ViT and ConvNeXt models, and the `efficientnet_pytorch` library for EfficientNet. Ensure you have these libraries installed:

```
pip install efficientnet_pytorch timm
```

## Number of parameters


| use_metadata\model | ConvNext | ViT | EfficientNet | ResNet50 | ResNet34 | 
|---|---|---|---|---|---|
| false | 87.57M | 85.8M | 4.02M | 23.52M | 21.29M
| true  | 87.64M | 85.86M | 4.1M | 23.65M | 21.33M


## Inference speed benchmark

Benchmark was done on NVIDIA GeForce RTX 4070 Super with batch size 128 and 10 workers.

| use_metadata\model | ConvNext | ViT | EfficientNet | ResNet50 | ResNet34 | 
|---|---|---|---|---|---|
| false | 403 | 456 | 632 | 640 | 893
| true  | 394 | 468 | 678 | 622 | 879



## Convergence time


| model | ConvNext | ViT | EfficientNet | ResNet50 | ResNet34 | 
|---|---|---|---|---|---|
| time | 1h 50min | 1h 6min | 33min | 37min | 21min
