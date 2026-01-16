import argparse
import os
import time
from typing import Iterable, Optional

from compression_core import compress_sq, detect_model_name


def default_output(model: str, n_bits: int) -> str:
    return os.path.join('compressed_models', f"{model}_per_input_channel_cifar100_{n_bits}bit.pth")


def sweep_bits() -> Iterable[int]:
    return list(range(3, 9))  # 3 -> 8 bits


def _to_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Scalar Quantization sweep for CIFAR-100")
    parser.add_argument('arg1', nargs='?', help='Either bits (int) or weights path')
    parser.add_argument('arg2', nargs='?', help='Optional weights path if arg1 was bits')
    parser.add_argument('--model', help='Model name (matches constructor under models/). If omitted, auto-detect from weights.')
    args = parser.parse_args()

    # Interpret positional args flexibly:
    # - If arg1 is int, treat as bits and arg2 (if provided) as weights.
    # - Else, arg1 is weights and arg2 (if int) can override bits.
    bits = None
    weights = 'model.pth'

    first_int = _to_int(args.arg1)
    if first_int is not None:
        bits = first_int
        weights = args.arg2 or 'model.pth'
    else:
        if args.arg1:
            weights = args.arg1
        bits = _to_int(args.arg2)

    model_name = args.model or detect_model_name(weights)
    bits_list = [bits] if bits is not None else list(sweep_bits())

    for b in bits_list:
        compressed_path = default_output(model_name, b)
        t0 = time.time()
        top1 = compress_sq(model_name, weights, b, compressed_path)
        t1 = time.time()
        print(f"Saved: {compressed_path}")
        print(f"CIFAR-100 Top-1: {top1:.3f}% | bits={b} | time={t1 - t0:.2f}s")


if __name__ == '__main__':
    main()