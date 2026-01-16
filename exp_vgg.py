"""CIFAR-100 evaluation utilities with dynamic model loading.

This module now focuses solely on CIFAR-100 validation. Models are
resolved dynamically from the ``models`` package by name so you can
plug in any architecture you trained on CIFAR-100.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
from typing import Callable, Dict

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class AverageMeter:
    """Track a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def accuracy(output, target, topk=(1,)):
    """Compute precision@k."""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a leading ``module.`` prefix introduced by DataParallel."""

    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def _iter_model_builders():
    """Yield (name, callable) pairs for all symbols in ``models`` modules."""

    package = importlib.import_module('models')
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f'models.{module_name}')
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr):
                yield attr_name, attr


def resolve_model_builder(model_name: str, num_classes: int = 100) -> Callable[[], nn.Module]:
    """Return a constructor for the requested model name.

    The search scans every module under ``models`` and returns the first
    callable whose attribute name matches ``model_name``.
    """

    available: Dict[str, Callable] = {}
    for attr_name, builder in _iter_model_builders():
        available[attr_name] = builder

    def _wrap(builder):
        def _builder():
            try:
                return builder(num_classes=num_classes)
            except TypeError:
                return builder()
        return _builder

    # Exact match first
    if model_name in available:
        return _wrap(available[model_name])

    # Try simple BN/non-BN fallback for VGG-style names
    if model_name.endswith('_bn') and model_name[:-3] in available:
        return _wrap(available[model_name[:-3]])
    if (model_name + '_bn') in available:
        return _wrap(available[model_name + '_bn'])

    raise ValueError(f"Model '{model_name}' not found under models/. Available: {sorted(available.keys())[:20]}...")


def build_cifar100_loader(batch_size: int = 128, workers: int = 4):
    """Create a CIFAR-100 validation dataloader."""

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return val_loader


def validate_cifar100(model_path: str, model_name: str, batch_size: int = 128,
                      workers: int = 4, half: bool = False, cpu: bool = False,
                      print_freq: int = 10):
    """Load a checkpoint, run CIFAR-100 validation, and return top-1 accuracy."""

    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    builder = resolve_model_builder(model_name)
    model = builder()
    try:
        model.load_state_dict(state_dict)
    except Exception:
        try:
            model.load_state_dict(strip_module_prefix(state_dict))
        except Exception as e:
            raise RuntimeError(
                "Failed to load checkpoint weights. Please ensure the checkpoint matches a CIFAR-100 model with 100 classes.\n"
                "If you are using a CIFAR-10 checkpoint, switch to a CIFAR-100 trained checkpoint (and ensure the architecture name matches constructors under models/)."
            ) from e

    model.to(device)
    if half:
        model.half()

    criterion = nn.CrossEntropyLoss().to(device)
    val_loader = build_cifar100_loader(batch_size=batch_size, workers=workers)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if half:
            images = images.half()

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                f"Test: [{i}/{len(val_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
            )

    print(f" * Prec@1 {top1.avg:.3f}")
    return top1.avg


__all__ = [
    'validate_cifar100',
    'resolve_model_builder',
    'strip_module_prefix',
    'AverageMeter',
    'accuracy',
]
