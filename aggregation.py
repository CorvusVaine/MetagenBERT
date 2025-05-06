import argparse
import numpy as np
import os
from joblib import Parallel, delayed
import torch

def load_and_compute_partial_sum(file_path):
    """Load a file and return the sum and count of arrays."""
    data = torch.load(file_path).cpu().numpy()  # Load efficiently
    return np.sum(data, axis=0), data.shape[0]  # Return sum and count

def compute_weighted_mean(directory, num_workers=24):
    """Compute the overall mean, considering different array counts."""
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
    # Parallel computation to process multiple files efficiently
    results = Parallel(n_jobs=num_workers)(delayed(load_and_compute_partial_sum)(fp) for fp in file_paths)
    print(len(results))
    # Aggregate sums and counts
    total_sum = np.zeros_like(results[0][0])  # Initialize sum array
    total_count = 0

    for partial_sum, count in results:
        total_sum += partial_sum
        total_count += count

    # Compute weighted mean
    return total_sum / total_count if total_count > 0 else None

def main(directory,save_dir):
    L=os.listdir(directory)
    sorted(L)
    for file in L:
        print(file)
        mean = compute_weighted_mean(os.path.join(directory,file))
        np.save(os.path.join(save_dir,file+".npy"),mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_dir", type=str, help="Path to the directory of samples")
    parser.add_argument("save_path", type=str, help="Path to save the aggregated embeddings")
    args = parser.parse_args()
    main(args.samples_dir, args.save_path)

