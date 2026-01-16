import argparse
import os
import time

from compression_core import compress_pq, detect_model_name


def default_output(model: str, nsubq: int, nbits: int) -> str:
    ncentroids = 2 ** nbits
    return os.path.join('compressed_models', f"{model}_faiss_pq_cifar100_{nsubq}subq_{nbits}bits_{ncentroids}cent.pth")


def main():
    parser = argparse.ArgumentParser(description="FAISS PQ (short) for CIFAR-100")
    parser.add_argument('nsubq', type=int, help='Number of subspaces')
    parser.add_argument('nbits', type=int, help='Bits per subspace (<=24)')
    parser.add_argument('weights', nargs='?', default='model.pth', help='Weights path (default: model.pth)')
    parser.add_argument('--output', help='Output path for compressed checkpoint')
    args = parser.parse_args()

    model = detect_model_name(args.weights)
    compressed_path = args.output or default_output(model, args.nsubq, args.nbits)

    t0 = time.time()
    top1 = compress_pq(None, args.weights, args.nsubq, args.nbits, compressed_path)
    t1 = time.time()

    print(f"Saved: {compressed_path}")
    print(f"CIFAR-100 Top-1: {top1:.3f}%")
    print(f"Time: {t1 - t0:.2f}s")


if __name__ == '__main__':
    main()