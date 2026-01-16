import argparse
import os
import time
from typing import Iterable, Optional

from compression_core import compress_prune, detect_model_name


def default_output(model: str, amount: float) -> str:
    return os.path.join('compressed_models', f"{model}_pytorch_pruning_cifar100_{amount:.2f}.pth")


def sweep_amounts() -> Iterable[float]:
    return [round(x / 10.0, 2) for x in range(1, 10)]  # 0.1 -> 0.9


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Pruning sweep for CIFAR-100")
    parser.add_argument('arg1', nargs='?', help='Either amount (float) or weights path')
    parser.add_argument('arg2', nargs='?', help='Optional weights path if arg1 was amount')
    parser.add_argument('--model', help='Model name (matches constructor under models/). If omitted, auto-detect from weights.')
    args = parser.parse_args()

    # Interpret positional args flexibly:
    # - If arg1 is float, treat as amount and arg2 (if provided) as weights.
    # - Else, arg1 is weights and arg2 (if float) can override amount.
    amount = None
    weights = 'model.pth'

    first_float = _to_float(args.arg1)
    if first_float is not None:
        amount = first_float
        weights = args.arg2 or 'model.pth'
    else:
        if args.arg1:
            weights = args.arg1
        amount = _to_float(args.arg2)

    model_name = args.model or detect_model_name(weights)
    amounts = [amount] if amount is not None else list(sweep_amounts())

    for amt in amounts:
        compressed_path = default_output(model_name, amt)
        t0 = time.time()
        top1 = compress_prune(model_name, weights, amt, compressed_path)
        t1 = time.time()
        print(f"Saved: {compressed_path}")
        print(f"CIFAR-100 Top-1: {top1:.3f}% | amount={amt:.2f} | time={t1 - t0:.2f}s")


if __name__ == '__main__':
    main()