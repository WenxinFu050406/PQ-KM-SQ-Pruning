"""Shared compression utilities targeting CIFAR-100 models."""

from __future__ import annotations

import os
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set

import faiss
import torch

from exp_vgg import validate_cifar100


@dataclass
class FlattenedWeights:
    flat: torch.Tensor
    conv_keys: List[str]
    # Shapes can be 4-D (standard conv) or 3-D (depthwise weights saved as [C, 3, 3])
    shapes: Dict[str, Tuple[int, ...]]


def load_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def detect_model_name(model_path: str) -> str:
    """Best-effort model architecture detection from a checkpoint.

    Priority:
    1) Read metadata keys 'arch'/'model'/'model_name' (including meta dict).
    2) Match the weight filename against callable names discovered under models/.

    If neither yields a match, raise an error instead of guessing (VGG heuristic removed).
    For reliable behavior with non-VGG models, provide metadata when saving or pass the
    model name explicitly on the CLI.
    """

    def _available_model_names() -> Set[str]:
        """Collect callable names under models/ that look like factory functions.

        We deliberately keep lowercase names to avoid matching base classes like VGG
        that require positional args and cannot be auto-instantiated.
        """
        names: Set[str] = set()
        try:
            package = importlib.import_module('models')
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                module = importlib.import_module(f'models.{module_name}')
                for attr_name in dir(module):
                    if not attr_name.islower():
                        continue
                    attr = getattr(module, attr_name)
                    if callable(attr):
                        names.add(attr_name)
        except Exception:
            pass
        return names

    available_names = _available_model_names()

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Try metadata first
    def _from_meta(obj: Dict[str, torch.Tensor]) -> str | None:
        for key in ('arch', 'model', 'model_name'):
            val = obj.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    if isinstance(checkpoint, dict):
        meta_hit = _from_meta(checkpoint)
        if meta_hit:
            return meta_hit
        meta = checkpoint.get('meta') if isinstance(checkpoint.get('meta'), dict) else None
        if meta:
            meta_hit = _from_meta(meta)
            if meta_hit:
                return meta_hit

    # Filename clue
    basename = os.path.basename(model_path).lower()
    matched = None
    if available_names:
        candidates = [name for name in available_names if name.lower() in basename]
        if candidates:
            # pick the longest match to avoid partial hits (e.g., resnet vs resnet50)
            matched = sorted(candidates, key=len, reverse=True)[0]
    if matched:
        return matched

    raise RuntimeError(
        "Model name could not be inferred from checkpoint metadata or filename. "
        "Add 'arch'/'model'/'model_name' to the checkpoint, ensure the filename contains a known builder name, "
        "or pass --model_name explicitly when running compression."
    )


def flatten_conv_weights(state_dict: Dict[str, torch.Tensor]) -> FlattenedWeights:
    conv_keys: List[str] = []
    shapes: Dict[str, Tuple[int, ...]] = {}
    flat_tensors: List[torch.Tensor] = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        # Standard conv [out, in, kH, kW]
        if v.ndim == 4 and v.shape[2] == 3 and v.shape[3] == 3:
            conv_keys.append(k)
            shapes[k] = tuple(v.shape)
            flat_tensors.append(v.reshape(v.shape[0] * v.shape[1], -1))
        # Depthwise-exported weights saved as [C, 3, 3]
        elif v.ndim == 3 and v.shape[-2:] == (3, 3):
            conv_keys.append(k)
            shapes[k] = tuple(v.shape)
            flat_tensors.append(v.reshape(v.shape[0], -1))
        else:
            continue
    if not flat_tensors:
        raise ValueError("No convolutional weights found to compress.")
    flat = torch.cat(flat_tensors, dim=0)
    return FlattenedWeights(flat=flat, conv_keys=conv_keys, shapes=shapes)


def rebuild_state_dict(flat: torch.Tensor, original: Dict[str, torch.Tensor], info: FlattenedWeights) -> Dict[str, torch.Tensor]:
    rebuilt = original.copy()
    offset = 0
    for key in info.conv_keys:
        shape = info.shapes[key]
        if len(shape) == 4:
            a, b, c, d = shape
            size = a * b
            chunk = flat[offset: offset + size].reshape(a, b, c, d)
            offset += size
        elif len(shape) == 3:
            a, c, d = shape
            size = a
            chunk = flat[offset: offset + size].reshape(a, c, d)
            offset += size
        else:
            raise ValueError(f"Unsupported conv weight shape {shape} for key {key}")
        rebuilt[key] = chunk
    return rebuilt


def save_compressed_state(state_dict: Dict[str, torch.Tensor], output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({'state_dict': state_dict}, output_path)


def compress_kmeans(model_name: str | None, model_path: str, num_clusters: int, output_path: str) -> float:
    if not model_name:
        model_name = detect_model_name(model_path)
    state_dict = load_state_dict(model_path)
    weights = flatten_conv_weights(state_dict)

    kernel_np = weights.flat.cpu().numpy().astype('float32')
    d = kernel_np.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True)
    kmeans.train(kernel_np)
    _, cluster_ids = kmeans.index.search(kernel_np, 1)
    cluster_ids = cluster_ids.reshape(-1)
    centroids = torch.from_numpy(kmeans.centroids).to(weights.flat.device).type(weights.flat.dtype)
    compressed_flat = centroids[cluster_ids]

    rebuilt = rebuild_state_dict(compressed_flat, state_dict, weights)
    save_compressed_state(rebuilt, output_path)
    return float(validate_cifar100(output_path, model_name))


def compress_pq(model_name: str | None, model_path: str, nsubq: int, nbits: int, output_path: str) -> float:
    if not model_name:
        model_name = detect_model_name(model_path)
    state_dict = load_state_dict(model_path)
    weights = flatten_conv_weights(state_dict)

    kernel_np = weights.flat.cpu().numpy().astype('float32')
    dim = kernel_np.shape[1]
    if dim % nsubq != 0:
        raise ValueError(f"Subspace count {nsubq} must divide kernel dimension {dim}.")
    pq = faiss.ProductQuantizer(dim, nsubq, nbits)
    pq.train(kernel_np)
    codes = pq.compute_codes(kernel_np)
    decoded = pq.decode(codes)
    compressed_flat = torch.from_numpy(decoded).to(weights.flat.device).type(weights.flat.dtype)

    rebuilt = rebuild_state_dict(compressed_flat, state_dict, weights)
    save_compressed_state(rebuilt, output_path)
    return float(validate_cifar100(output_path, model_name))


def compress_prune(model_name: str | None, model_path: str, pruning_amount: float, output_path: str) -> float:
    if pruning_amount < 0 or pruning_amount > 1:
        raise ValueError("pruning_amount must be in [0, 1]")
    if not model_name:
        model_name = detect_model_name(model_path)

    state_dict = load_state_dict(model_path)
    weights = flatten_conv_weights(state_dict)
    flat = weights.flat

    all_weights = flat.abs().flatten()
    k = int(pruning_amount * all_weights.numel())
    if k > 0:
        threshold = torch.kthvalue(all_weights, k).values
        mask = (flat.abs() > threshold).float()
        pruned_flat = flat * mask
    else:
        pruned_flat = flat.clone()

    rebuilt = rebuild_state_dict(pruned_flat, state_dict, weights)
    save_compressed_state(rebuilt, output_path)
    return float(validate_cifar100(output_path, model_name))


def compress_sq(model_name: str | None, model_path: str, n_bits: int, output_path: str) -> float:
    if not model_name:
        model_name = detect_model_name(model_path)
    state_dict = load_state_dict(model_path)

    def quantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / (2 ** (n_bits - 1) - 1) if max_val > 0 else 1.0
        quantized = torch.clamp(torch.round(tensor / scale), -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
        return quantized * scale

    quantized_state: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        # Quantize only 3x3 conv kernels. This covers standard 4D conv weights
        # and exported depthwise/per-channel 3D weights shaped [C, 3, 3].
        if (tensor.ndim == 4 and tensor.shape[-2:] == (3, 3)) or (tensor.ndim == 3 and tensor.shape[-2:] == (3, 3)):
            quantized_state[name] = quantize_tensor(tensor)
        else:
            quantized_state[name] = tensor

    save_compressed_state(quantized_state, output_path)
    return float(validate_cifar100(output_path, model_name))


__all__ = [
    'compress_kmeans',
    'compress_pq',
    'compress_prune',
    'compress_sq',
    'detect_model_name',
    'load_state_dict',
    'flatten_conv_weights',
    'rebuild_state_dict',
]
