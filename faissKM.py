import argparse
import os
import time

from compression_core import compress_kmeans, detect_model_name


def default_output(model: str, clusters: int) -> str:
    return os.path.join('compressed_models', f"{model}_faiss_kmeans_cifar100_{clusters}.pth")


def main():
    parser = argparse.ArgumentParser(description="FAISS k-means (short) for CIFAR-100")
    parser.add_argument('clusters', type=int, help='Number of k-means clusters')
    parser.add_argument('weights', nargs='?', default='model.pth', help='Weights path (default: model.pth)')
    parser.add_argument('--output', help='Output path for compressed checkpoint')
    args = parser.parse_args()

    model = detect_model_name(args.weights)
    compressed_path = args.output or default_output(model, args.clusters)

    t0 = time.time()
    top1 = compress_kmeans(None, args.weights, args.clusters, compressed_path)
    t1 = time.time()

    print(f"Saved: {compressed_path}")
    print(f"CIFAR-100 Top-1: {top1:.3f}%")
    print(f"Time: {t1 - t0:.2f}s")


if __name__ == '__main__':
    main()