"""
Part 1 - Data exploration and baseline analysis
Tasks: 1 (visualisation), 2 (timing), 3 (simulation results), 4 (profiling)
"""

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from simulate import load_data, jacobi, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4


def task1_visualize_input(building_ids, n=4):
    """Task 1: Visualise input domain and interior mask for n buildings."""
    print("=" * 60)
    print("TASK 1 - Input data visualisation")
    print("=" * 60)

    fig, axes = plt.subplots(2, n, figsize=(4 * n + 2, 8))
    fig.suptitle("Input data: domain and interior masks", y = 0.98)

    for idx in range(n):
        bid = building_ids[idx]
        u0, mask = load_data(LOAD_DIR, bid)

        # Top row: initial temperature grid
        im = axes[0, idx].imshow(u0, cmap='hot', vmin=0, vmax=25)
        axes[0, idx].set_title(f'Domain\nID: {bid}', fontsize=9)
        axes[0, idx].axis('off')

        # Bottom row: interior mask
        axes[1, idx].imshow(mask, cmap='gray')
        axes[1, idx].set_title(f'Interior mask\nID: {bid}', fontsize=9)
        axes[1, idx].axis('off')

    fig.colorbar(im, ax=axes[:, :], location='right', label='Temperature (°C)', fraction=0.015, pad=0.02)

    #plt.tight_layout()
    plt.savefig('part1_task1_input_data.png', dpi=300, bbox_inches='tight')
    print("  Saved: part1_task1_input_data.png")
    plt.close()


def task2_timing(building_ids, n_test=10):
    """Task 2: Time the reference implementation and extrapolate to full dataset."""
    print("\n" + "=" * 60)
    print(f"TASK 2 - Timing on {n_test} buildings")
    print("=" * 60)

    ids_test = building_ids[:n_test]
    n_total  = 4571
    t_load_list = []
    t_sim_list = []
    for _ in range(15):
        t0 = time.perf_counter()
        all_u0, all_masks = [], []
        for bid in ids_test:
            u0, mask = load_data(LOAD_DIR, bid)
            all_u0.append(u0)
            all_masks.append(mask)
        t_load = time.perf_counter() - t0
        t_load_list.append(t_load)

        t0 = time.perf_counter()
        for u0, mask in zip(all_u0, all_masks):
            jacobi(u0, mask, MAX_ITER, ABS_TOL)
        t_sim = time.perf_counter() - t0
        t_sim_list.append(t_sim)
    t_sim = np.mean(t_sim_list)
    t_load = np.mean(t_load_list)

    t_total = t_load + t_sim
    t_extrap = (t_total / n_test) * n_total
    print(f"  With 15 iterations the average was:")
    print(f"  Loading    : {t_load:.2f}s  ({n_test} buildings)")
    print(f"  Simulation : {t_sim:.2f}s  ({n_test} buildings)")
    print(f"  Total      : {t_total:.2f}s  ({n_test} buildings)")
    print(f"  Per building: {t_total/n_test:.2f}s")
    print(f"\n  Estimated for {n_total} buildings: {t_extrap/60:.1f} min "
          f"({t_extrap/3600:.2f} h)")

    return t_total / n_test


def task3_visualize_results(building_ids, n=4):
    """Task 3: Run simulation and visualise steady-state temperature fields."""
    print("\n" + "=" * 60)
    print("TASK 3 - Simulation results visualisation")
    print("=" * 60)

    fig, axes = plt.subplots(1, n, figsize=(4 * n + 3, 5))
    fig.suptitle("Simulation results: steady-state temperature distribution", fontsize=14)

    for idx in range(n):
        bid = building_ids[idx]
        u0, mask = load_data(LOAD_DIR, bid)

        print(f"  Simulating building {bid} ...", end=' ', flush=True)
        t0 = time.perf_counter()
        u_result = jacobi(u0, mask, MAX_ITER, ABS_TOL)
        print(f"{time.perf_counter()-t0:.1f}s")

        display = u_result.copy()
        display[display == 0] = np.nan

        im = axes[idx].imshow(display, cmap='inferno', vmin=5, vmax=25)
        axes[idx].set_title(f'ID: {bid}', fontsize=9)
        axes[idx].axis('off')

        stats = summary_stats(u_result, mask)
        info  = (f"mean={stats['mean_temp']:.1f}°C\n"
                 f">18°C: {stats['pct_above_18']:.0f}%")
        axes[idx].text(0.02, 0.02, info, transform=axes[idx].transAxes,
               fontsize=13, color='white',
               bbox=dict(facecolor='black', alpha=0.5, pad=10))

    plt.subplots_adjust(right=0.88) #AI begin
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height] - AI end
    fig.colorbar(im, cax=cbar_ax, label='Temperature (°C)' , fraction=0.015, pad=0.02)
    plt.savefig('part1_task3_simulation_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: part1_task3_simulation_results.png")
    plt.close()


if __name__ == '__main__':
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print(f"Dataset: {len(building_ids)} buildings available.")
    print(f"Sample IDs: {building_ids[:5]}")

    task1_visualize_input(building_ids, n=4)
    task2_timing(building_ids, n_test=10)
    task3_visualize_results(building_ids, n=4)

    print("\nPart 1 complete.")
