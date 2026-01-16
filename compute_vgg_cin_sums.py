import torch
import torchvision.models as models


def sum_conv_cin(model):
    total_cin = 0
    per_layer = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            total_cin += module.in_channels
            per_layer.append((name, module.in_channels))
    return total_cin, per_layer


def main():
    vgg_names = ["vgg11", "vgg16", "vgg19"]
    for name in vgg_names:
        # instantiate without pretrained weights
        constructor = getattr(models, name)
        model = constructor(weights=None)
        total_cin, per_layer = sum_conv_cin(model)
        print(f"{name}: total cin = {total_cin}")
        # Uncomment below to see per-layer cin details
        # for layer_name, cin in per_layer:
        #     print(f"  {layer_name}: cin={cin}")


if __name__ == "__main__":
    main()
