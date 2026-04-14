"""
Task 8
"""

from os.path import join
import sys
import time

import numpy as np
from numba import cuda

from simulate import load_data, jacobi as jacobi_reference, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    rows, cols = interior_mask.shape
    if 1 <= i <= rows and 1 <= j <= cols:
        if interior_mask[i-1, j-1]:
            u_new[i,j] = 0.25*(u[i,j-1]+u[i,j+1]
                               +u[i-1,j]+u[i+1,j])
        else:
            u_new[i, j] = u[i, j]

def helper(u, interior_mask, max_iter):
    u_new = np.zeros(u.shape, dtype=u.dtype)
    u, u_new, interior_mask = cuda.to_device(u), cuda.to_device(u_new), cuda.to_device(interior_mask) # To GPU
    rows, cols = u.shape
    tpb = (32,32) # Threads per block
    bpg = (rows // tpb[0]+1, cols // tpb[1]+1) # Blocks per grid
    for _ in range(max_iter):
        jacobi_kernel[bpg,tpb](u, u_new, interior_mask)
        u, u_new = u_new, u # swapping so u it becomes the newly computed u_new, 
                            # and reusing memory of old u to write onto it again
    return u.copy_to_host() # to CPU


def warmup(u_sample, mask_sample):
    """Trigger JIT compilation before benchmarking."""
    print("  JIT warm-up...", end=' ', flush=True)
    t0 = time.perf_counter()
    _ = helper(u_sample.copy(), mask_sample, 1)
    print(f"done in {time.perf_counter()-t0:.1f}s")


def benchmark(building_ids, n_test):
    """Benchmark reference NumPy vs custom CUDA."""
    print(f"\n{'Method':<25} | {'Time (s)':>10} | {'Speed-up':>10}")
    print("-" * 52)

    ids = building_ids[:n_test]

    # (a) Reference NumPy
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
    t_ref = time.perf_counter() - t0
    print(f"{'(a) Reference NumPy':<25} | {t_ref:>10.2f} | {'1.00x':>10}")

    # Warm-up JIT
    u_sample, mask_sample = load_data(LOAD_DIR, ids[0])
    warmup(u_sample, mask_sample)

    # (b) cuda
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = helper(u0,mask, max_iter = MAX_ITER)
    t_seq = time.perf_counter() - t0
    print(f"{'(b) CUDA JIT':<25} | {t_seq:>10.2f} | {t_ref/t_seq:>9.2f}x")

    print(f"\n  Extrapolation to {N_TOTAL} buildings:")
    for label, t in [('Reference', t_ref), ('Cuda JIT', t_seq)]:
        t_est = (t / n_test) * N_TOTAL
        print(f"    {label:<12}: {t_est/60:.1f} min ({t_est/3600:.2f}h)")


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print("=" * 55)
    print("  Task 8 - custom CUDA implementation")
    print("=" * 55)
    print(f"  Test buildings: {N}")

    benchmark(building_ids, n_test=N)
