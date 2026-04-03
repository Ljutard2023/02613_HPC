"""
Part 2 - CPU parallelisation (static and dynamic scheduling)
Tasks: 5 (static scheduling + speed-up plots + Amdahl's law)
       6 (dynamic scheduling)

Usage:
    python part2_parallel.py <N> <mode>
    e.g.: python part2_parallel.py 100 static
          python part2_parallel.py 100 dynamic
          python part2_parallel.py 100 all
"""

from os.path import join
import sys
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from simulate import load_data, jacobi, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571


def process_building(bid):
    """Load, simulate and return summary stats for one building.
    Must be defined at module level for multiprocessing pickle."""
    u0, mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, mask, MAX_ITER, ABS_TOL)
    return bid, summary_stats(u, mask)


def run_static(building_ids, n_workers):
    """Static scheduling: pool.map() divides work into equal chunks upfront."""
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_building, building_ids)
    return results


def run_dynamic(building_ids, n_workers):
    """Dynamic scheduling: each worker picks one building at a time (chunksize=1)."""
    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(process_building, building_ids, chunksize=1))
    return results


def measure_speedup(building_ids, max_workers=None, mode='static'):
    """Measure wall time for increasing worker counts and compute speed-ups."""
    if max_workers is None:
        max_workers = mp.cpu_count()

    worker_counts = [w for w in [1, 2, 4, 8, 16, 32] if w <= max_workers]
    times, speedups = [], []
    t_ref = None

    print(f"\n{'Workers':>8} | {'Time (s)':>10} | {'Speed-up':>10}")
    print("-" * 35)

    for n_w in worker_counts:
        t0 = time.perf_counter()
        if mode == 'static':
            run_static(building_ids, n_w)
        else:
            run_dynamic(building_ids, n_w)
        t = time.perf_counter() - t0

        times.append(t)
        if t_ref is None:
            t_ref = t
        s = t_ref / t
        speedups.append(s)
        print(f"{n_w:>8} | {t:>10.2f} | {s:>10.2f}x")

    return worker_counts, times, speedups


def amdahl_analysis(worker_counts, speedups):
    """Estimate parallel fraction p from Amdahl's law: S(N) = 1/((1-p) + p/N)."""
    print("\n--- Amdahl's law analysis ---")
    N_best = worker_counts[-1]
    S_best = speedups[-1]

    p = max(0.0, min(1.0, (1/S_best - 1) / (1/N_best - 1))) if N_best > 1 else 0.0
    S_max = 1 / (1 - p) if p < 1 else float('inf')

    print(f"  Parallel fraction p       : {p*100:.1f}%")
    print(f"  Sequential fraction (1-p) : {(1-p)*100:.1f}%")
    print(f"  Theoretical max speed-up  : {S_max:.1f}x")
    print(f"  Achieved with {N_best} workers  : {S_best:.2f}x  "
          f"({S_best/S_max*100:.0f}% of theoretical max)")
    return p, S_max


def plot_speedups(wc_s, sp_s, wc_d, sp_d, p_s, p_d, n_buildings):
    """Plot measured speed-up vs ideal and Amdahl curves for both schedules."""
    N_range = np.linspace(1, max(max(wc_s), max(wc_d)), 200)
    amdahl  = lambda p, n: 1 / ((1 - p) + p / n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Speed-up — {n_buildings} buildings', fontsize=13)

    for ax, label, wc, sp, p in [
        (axes[0], 'Static',  wc_s, sp_s, p_s),
        (axes[1], 'Dynamic', wc_d, sp_d, p_d),
    ]:
        ax.plot(wc, sp, 'o-', color='royalblue', lw=2, label='Measured speed-up')
        ax.plot(N_range, N_range, '--', color='gray', lw=1.5, label='Ideal (linear)')
        ax.plot(N_range, amdahl(p, N_range), ':', color='tomato', lw=2,
                label=f'Amdahl (p={p*100:.0f}%)')
        ax.set_xlabel('Number of workers (CPU cores)')
        ax.set_ylabel('Speed-up S(N)')
        ax.set_title(f'{label} scheduling')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=1)

    plt.tight_layout()
    plt.savefig('part2_speedup.png', dpi=150, bbox_inches='tight')
    print("  Saved: part2_speedup.png")
    plt.close()


if __name__ == '__main__':
    N    = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    mode = sys.argv[2]      if len(sys.argv) > 2 else 'all'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    max_w = mp.cpu_count()
    print(f"Available CPUs : {max_w}")
    print(f"Buildings used : {N}")

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    if mode in ('static', 'all'):
        print("\n=== STATIC SCHEDULING ===")
        wc_s, t_s, sp_s = measure_speedup(building_ids, max_w, mode='static')
        p_s, _ = amdahl_analysis(wc_s, sp_s)
        best_t_s = min(t_s)
        print(f"  Best: {best_t_s:.2f}s ({N} buildings) -> "
              f"{(best_t_s/N)*N_TOTAL/60:.1f} min for {N_TOTAL} buildings")

    if mode in ('dynamic', 'all'):
        print("\n=== DYNAMIC SCHEDULING ===")
        wc_d, t_d, sp_d = measure_speedup(building_ids, max_w, mode='dynamic')
        p_d, _ = amdahl_analysis(wc_d, sp_d)
        best_t_d = min(t_d)
        best_w_d = wc_d[t_d.index(best_t_d)]
        print(f"  Best: {best_t_d:.2f}s ({N} buildings) -> "
              f"{(best_t_d/N)*N_TOTAL/60:.1f} min for {N_TOTAL} buildings")

    if mode == 'all':
        plot_speedups(wc_s, sp_s, wc_d, sp_d, p_s, p_d, N)
        results = run_dynamic(building_ids, best_w_d)
        print('building_id,' + ','.join(stat_keys))
        for bid, stats in results:
            print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
