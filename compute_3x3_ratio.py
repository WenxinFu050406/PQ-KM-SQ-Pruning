"""Compute proportion of parameters coming from 1x1, 3x3 and 5x5 convolution weights for each supported network.

Numerators: sum of numel of Conv2d.weight where kernel_size == (1,1)/(3,3)/(5,5) (or kernel_size==1/3/5)
Denominator: total number of trainable parameters (sum of numel of p for p.requires_grad)

Run from project root: python .\compute_3x3_ratio.py
"""
import torch
import torch.nn as nn
from types import SimpleNamespace
import utils
import math

# list of network names supported by utils.get_network
net_names = [
    'vgg16','vgg13','vgg11','vgg19',
    'densenet121','densenet161','densenet169','densenet201',
    'googlenet',
    'inceptionv3','inceptionv4','inceptionresnetv2',
    'xception',
    'resnet18','resnet34','resnet50','resnet101','resnet152',
    'preactresnet18','preactresnet34','preactresnet50','preactresnet101','preactresnet152',
    'resnext50','resnext101','resnext152',
    'shufflenet','shufflenetv2',
    'squeezenet',
    'mobilenet','mobilenetv2',
    'nasnet',
    'attention56','attention92',
    'seresnet18','seresnet34','seresnet50','seresnet101','seresnet152',
    'wideresnet',
    'stochasticdepth18','stochasticdepth34','stochasticdepth50','stochasticdepth101'
]

results = []
for name in net_names:
    args = SimpleNamespace(net=name, gpu=False)
    try:
        net = utils.get_network(args)
    except Exception as e:
        results.append((name, None, None, None, None, str(e)))
        continue

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    params_1x1 = 0
    params_3x3 = 0
    params_5x5 = 0

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            ks = m.kernel_size
            k = None
            if isinstance(ks, tuple) and len(ks) >= 1:
                k = ks[0]
            elif isinstance(ks, int):
                k = ks

            w = getattr(m, 'weight', None)

            if k == 1 and w is not None:
                params_1x1 += w.numel()
            elif k == 3 and w is not None:
                # Standard conv is [out,in,3,3]; depthwise-exported weights may appear as [C,3,3]
                params_3x3 += w.numel()
            elif k == 5 and w is not None:
                params_5x5 += w.numel()
            elif param.dim() == 3 and param.shape[-2:] == (3, 3):
                # Depthwise or per-channel exported conv weights saved as [C, 3, 3]
                params_3x3 += param.numel()

    ratio_1x1 = params_1x1 / total_params if total_params > 0 else 0
    ratio_3x3 = params_3x3 / total_params if total_params > 0 else 0
    ratio_5x5 = params_5x5 / total_params if total_params > 0 else 0

    results.append((name, total_params, params_1x1, ratio_1x1, params_3x3, ratio_3x3, params_5x5, ratio_5x5))

# print table
print('model,total_params,params_1x1,ratio_1x1,params_3x3,ratio_3x3,params_5x5,ratio_5x5')
for row in results:
    if row[1] is None:
        name = row[0]
        err = row[-1]
        print(f"{name},ERROR,0,0,0,0,0,0,\"{err}\"")
    else:
        name, total, p1, r1, p3, r3, p5, r5 = row
        print(f"{name},{total},{p1},{r1:.4f},{p3},{r3:.4f},{p5},{r5:.4f}")

# also save to file
with open('3x3_params_ratio.csv', 'w') as f:
    f.write('model,total_params,params_1x1,ratio_1x1,params_3x3,ratio_3x3,params_5x5,ratio_5x5\n')
    for row in results:
        if row[1] is None:
            name = row[0]
            err = row[-1]
            f.write(f"{name},ERROR,0,0,0,0,0,0,\"{err}\"\n")
        else:
            name, total, p1, r1, p3, r3, p5, r5 = row
            f.write(f"{name},{total},{p1},{r1:.6f},{p3},{r3:.6f},{p5},{r5:.6f}\n")

print('\nSaved results to 3x3_params_ratio.csv')
