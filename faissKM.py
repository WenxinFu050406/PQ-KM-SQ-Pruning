import argparse
import os
import sys
import time

from compression_core import compress_kmeans, detect_model_name


def default_output(model: str, clusters: int) -> str:
    return os.path.join('compressed_models', f"{model}_faiss_kmeans_cifar100_{clusters}.pth")


def main():
    # Manual parsing to support flexible argument order
    argv = sys.argv[1:]
    clusters_list = []
    weights_path = None
    output_path = None
    
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--output':
            if i + 1 < len(argv):
                output_path = argv[i + 1]
                i += 2
            else:
                print("Error: --output requires a value")
                sys.exit(1)
        elif arg.startswith('--'):
            print(f"Unknown option: {arg}")
            sys.exit(1)
        else:
            # Try to parse as integer (cluster value)
            try:
                clusters_list.append(int(arg))
                i += 1
            except ValueError:
                # Not an integer, treat as weights path
                if weights_path is None:
                    weights_path = arg
                    i += 1
                else:
                    print(f"Error: multiple weight paths specified ({weights_path} and {arg})")
                    sys.exit(1)
    
    # Set defaults
    if weights_path is None:
        weights_path = 'model.pth'
    
    if not clusters_list:
        print("Error: At least one cluster value must be specified")
        print("Usage: python faissKM.py <clusters...> [weights] [--output path]")
        print("Example: python faissKM.py 98304 wideresnet.pth")
        print("Example: python faissKM.py 65536 98304 131072 wideresnet.pth")
        sys.exit(1)

    model = detect_model_name(weights_path)
    
    for clusters in clusters_list:
        compressed_path = output_path or default_output(model, clusters)
        
        t0 = time.time()
        top1 = compress_kmeans(None, weights_path, clusters, compressed_path)
        t1 = time.time()

        print(f"Saved: {compressed_path}")
        print(f"CIFAR-100 Top-1: {top1:.3f}% | clusters={clusters} | time={t1 - t0:.2f}s")
        print()


if __name__ == '__main__':
    main()