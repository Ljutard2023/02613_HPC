"""
Part 3 - CPU optimisation with Numba JIT
Task: 7 (rewrite Jacobi using Numba JIT)

"""

from os.path import join
import sys
import time

import numpy as np
from numba import jit, prange

from simulate import load_data, jacobi as jacobi_reference, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571


@jit(nopython = True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-4):
    """
    Numba JIT sequential Jacobi solver.

    Converted the slicing to loops so the machine code is translated and cached in memory.
    """
    rows = interior_mask.shape[0]
    cols = interior_mask.shape[1]
    u_old = np.empty_like(u)

    for _ in range(max_iter):
        # Copy u -> u_old (preserve values while updating)
        for i in range(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        max_delta = 0.0

        for i in range(rows):
            for j in range(cols):
                if interior_mask[i, j]:
                    # Grid offset: interior index (i,j) maps to u[i+1, j+1]
                    val = 0.25 * (
                        u_old[i+1, j  ]   # left neighbour
                      + u_old[i+1, j+2]   # right neighbour
                      + u_old[i,   j+1]   # top neighbour
                      + u_old[i+2, j+1]   # bottom neighbour
                    )
                    diff = val - u_old[i+1, j+1]
                    if diff < 0.0:
                        diff = -diff     
                    if diff > max_delta:
                        max_delta = diff
                    u[i+1, j+1] = val

        if max_delta < atol:
            break

    return u


@jit(nopython = True, parallel=True)
def jacobi_numba_parallel(u, interior_mask, max_iter, atol=1e-4):
    """
    Same as other, but with parallel = True and prange instead of range
    """
    rows = interior_mask.shape[0]
    cols = interior_mask.shape[1]
    u_old = np.empty_like(u)

    for _ in range(max_iter):
        for i in prange(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        max_delta = 0.0
        for i in prange(rows):
            for j in range(cols):
                if interior_mask[i, j]:
                    val = 0.25 * (
                        u_old[i+1, j  ] + u_old[i+1, j+2]
                        + u_old[i,   j+1] + u_old[i+2, j+1]
                    )
                    diff = val - u_old[i+1, j+1]
                    if diff < 0.0:
                        diff = -diff
                    if diff > max_delta:
                        max_delta = diff
                    u[i+1, j+1] = val
        if max_delta < atol:
            break
    return u


def warmup_numba(u_sample, mask_sample):
    """Trigger JIT compilation before benchmarking."""
    print("  JIT warm-up...", end=' ', flush=True)
    t0 = time.perf_counter()
    _ = jacobi_numba(u_sample.copy(), mask_sample, max_iter=1)
    _ = jacobi_numba_parallel(u_sample.copy(), mask_sample, max_iter=1)
    print(f"done in {time.perf_counter()-t0:.1f}s")



def benchmark(building_ids, n_test):
    """Benchmark reference NumPy vs Numba sequential vs Numba parallel."""
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
    warmup_numba(u_sample, mask_sample)

    # (b) Numba sequential
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
    t_seq = time.perf_counter() - t0
    print(f"{'(b) Numba sequential':<25} | {t_seq:>10.2f} | {t_ref/t_seq:>9.2f}x")

    # (c) Numba parallel
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_numba_parallel(u0.copy(), mask, MAX_ITER, ABS_TOL)
    t_par = time.perf_counter() - t0
    print(f"{'(c) Numba parallel':<25} | {t_par:>10.2f} | {t_ref/t_par:>9.2f}x")

    print(f"\n  Extrapolation to {N_TOTAL} buildings:")
    for label, t in [('Reference', t_ref), ('Numba seq', t_seq), ('Numba par', t_par)]:
        t_est = (t / n_test) * N_TOTAL
        print(f"    {label:<12}: {t_est/60:.1f} min ({t_est/3600:.2f}h)")


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print("=" * 55)
    print("  Part 3 - Numba JIT optimisation")
    print("=" * 55)
    print(f"  Test buildings: {N}")

    benchmark(building_ids, n_test=N)
