"""
jacobi_profile.py - Line-by-line profiling script for the Jacobi function.

Usage:
    kernprof -l -v jacobi_profile.py

kernprof injects the @profile decorator and measures time spent on each line.
"""
from os.path import join
import numpy as np

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@profile  # special decorator injected by kernprof — do not import
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """Reference Jacobi function instrumented for line profiling."""
    u = np.copy(u)
    for i in range(max_iter):
        # Compute average of 4 neighbours for all interior points
        u_new = 0.25 * (
            u[1:-1, :-2]    # left neighbour
          + u[1:-1, 2:]     # right neighbour
          + u[:-2, 1:-1]    # top neighbour
          + u[2:, 1:-1]     # bottom neighbour
        )
        # Extract interior values via boolean mask
        u_new_interior = u_new[interior_mask]

        # Compute max change for convergence check
        delta = np.abs(
            u[1:-1, 1:-1][interior_mask] - u_new_interior
        ).max()

        # Update interior points only
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


if __name__ == '__main__':
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        bid = f.readline().strip()

    u0, interior_mask = load_data(LOAD_DIR, bid)

    print(f"Profiling jacobi on building {bid} (200 iterations)...")
    jacobi(u0, interior_mask, max_iter=200)
    print("Done.")
