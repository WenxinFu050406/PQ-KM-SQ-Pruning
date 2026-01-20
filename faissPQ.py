import argparse
import os
import sys
import time

from compression_core import compress_pq, detect_model_name


def default_output(model: str, nsubq: int, nbits: int) -> str:
    ncentroids = 2 ** nbits
    return os.path.join('compressed_models', f"{model}_faiss_pq_cifar100_{nsubq}subq_{nbits}bits_{ncentroids}cent.pth")


def main():
    # Manual parsing to support multiple (nsubq, nbits) pairs
    argv = sys.argv[1:]
    param_pairs = []
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
            # Try to parse as integer
            try:
                num1 = int(arg)
                # Need a second number for the pair
                if i + 1 < len(argv):
                    try:
                        num2 = int(argv[i + 1])
                        param_pairs.append((num1, num2))
                        i += 2
                        continue
                    except ValueError:
                        pass
                # Single number without pair - error
                print(f"Error: Parameter {num1} has no paired value")
                sys.exit(1)
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
    
    if not param_pairs:
        print("Error: At least one (nsubq, nbits) pair must be specified")
        print("Usage: python faissPQ.py <nsubq1> <nbits1> [<nsubq2> <nbits2> ...] [weights] [--output path]")
        print("Example: python faissPQ.py 3 15 model.pth")
        print("Example: python faissPQ.py 2 15 3 15 4 15 model.pth")
        sys.exit(1)

    model = detect_model_name(weights_path)
    
    for nsubq, nbits in param_pairs:
        compressed_path = output_path or default_output(model, nsubq, nbits)
        
        t0 = time.time()
        top1 = compress_pq(None, weights_path, nsubq, nbits, compressed_path)
        t1 = time.time()

        print(f"Saved: {compressed_path}")
        print(f"CIFAR-100 Top-1: {top1:.3f}% | nsubq={nsubq}, nbits={nbits} | time={t1 - t0:.2f}s")
        print()


if __name__ == '__main__':
    main()