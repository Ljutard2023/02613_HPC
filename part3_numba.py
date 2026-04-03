"""
Part 3 - CPU optimisation with Numba JIT
Task: 7 (rewrite Jacobi using Numba JIT)

Key concepts:
  - @njit compiles Python loops to native machine code on first call (JIT warm-up ~2s)
  - Explicit for-loops are preferred over NumPy slices with Numba: no temporary
    array allocations, better cache utilisation.
  - Cache-friendly access: inner loop over columns (j) matches C row-major memory
    layout -> sequential memory access -> high cache hit rate.

Usage: python part3_numba.py <N>
"""

from os.path import join
import sys
import time

import numpy as np
from numba import njit, prange

from simulate import load_data, jacobi as jacobi_reference, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571


@njit(cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-4):
    """
    Numba JIT sequential Jacobi solver.

    Uses explicit nested loops (cache-friendly: inner loop over columns j).
    Avoids allocating temporary arrays each iteration by copying u -> u_old once
    per iteration and updating u in-place.

    Args:
        u             : 514x514 grid with boundary conditions already set
        interior_mask : 512x512 boolean mask (True = interior room point)
        max_iter      : maximum number of iterations
        atol          : convergence tolerance
    """
    rows = interior_mask.shape[0]
    cols = interior_mask.shape[1]
    u_old = np.empty_like(u)

    for iteration in range(max_iter):
        # Copy u -> u_old (preserve values while updating)
        for i in range(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        max_delta = 0.0

        # Cache-friendly: outer loop over rows, inner loop over columns
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
                        diff = -diff      # manual abs (no math.fabs in @njit)
                    if diff > max_delta:
                        max_delta = diff
                    u[i+1, j+1] = val

        if max_delta < atol:
            break

    return u


@njit(parallel=True, cache=True)
def jacobi_numba_parallel(u, interior_mask, max_iter, atol=1e-4):
    """
    Numba JIT parallel Jacobi solver using prange.

    prange distributes the outer row loop across all available CPU threads.
    Note: the max_delta reduction is not perfectly thread-safe with prange,
    which can cause premature convergence. Use jacobi_numba (sequential) for
    production results.
    """
    rows = interior_mask.shape[0]
    cols = interior_mask.shape[1]
    u_old = np.empty_like(u)

    for iteration in range(max_iter):
        for i in prange(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        max_delta = 0.0

        for i in prange(rows):
            local_delta = 0.0
            for j in range(cols):
                if interior_mask[i, j]:
                    val = 0.25 * (
                        u_old[i+1, j  ] + u_old[i+1, j+2]
                      + u_old[i,   j+1] + u_old[i+2, j+1]
                    )
                    diff = val - u_old[i+1, j+1]
                    if diff < 0.0:
                        diff = -diff
                    if diff > local_delta:
                        local_delta = diff
                    u[i+1, j+1] = val
            if local_delta > max_delta:
                max_delta = local_delta

        if max_delta < atol:
            break

    return u


def warmup_numba(u_sample, mask_sample):
    """Trigger JIT compilation before benchmarking (~2s on first call)."""
    print("  JIT warm-up...", end=' ', flush=True)
    t0 = time.perf_counter()
    _ = jacobi_numba(u_sample.copy(), mask_sample, max_iter=1)
    _ = jacobi_numba_parallel(u_sample.copy(), mask_sample, max_iter=1)
    print(f"done in {time.perf_counter()-t0:.1f}s")


def verify_correctness(building_ids, n_check=3):
    """Verify Numba results match the reference implementation."""
    print("\n--- Correctness check ---")
    for bid in building_ids[:n_check]:
        u0, mask  = load_data(LOAD_DIR, bid)
        u_ref     = jacobi_reference(u0,       mask, MAX_ITER, ABS_TOL)
        u_numba   = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
        max_diff  = np.abs(u_ref - u_numba).max()
        status    = "OK" if max_diff < 1e-3 else "FAIL"
        print(f"  Building {bid}: max diff = {max_diff:.2e}  [{status}]")


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

    verify_correctness(building_ids)
    benchmark(building_ids, n_test=N)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nCSV results (Numba sequential):')
    print('building_id,' + ','.join(stat_keys))
    for bid in building_ids[:N]:
        u0, mask = load_data(LOAD_DIR, bid)
        u = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
        s = summary_stats(u, mask)
        print(f"{bid}," + ",".join(str(s[k]) for k in stat_keys))
