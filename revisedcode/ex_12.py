"""
Task: 12
"""
from os.path import join
import sys
import csv
import cupy as cp
from simulate import load_data, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571
def jacobi_cupy_opt(u, interior_mask, max_iter, stride=20, atol=1e-4):
    u = cp.asarray(u)
    interior_mask = cp.asarray(interior_mask)
    u_interior_view = u[1:-1, 1:-1] # We had memeory issues so AI suggested defining this
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_vals = u_new[interior_mask]
        
        if i % stride == 0: # Strided index so we check every 20 step.
            delta = cp.abs(u_interior_view[interior_mask] - u_new_vals).max()
            if delta < atol:
                break
        u_interior_view[interior_mask] = u_new_vals
        
    return cp.asnumpy(u)

if __name__ == '__main__':
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()


    u0, mask = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cupy_opt(u0, mask, max_iter=1)
    print("Warm-up OK", flush=True)

    print("=" * 55)
    print("  Exercise 12")
    print("=" * 55)


    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\n Optimised CuPy for all buildings....:')
    with open('results.csv', 'w', newline='') as f: 
        writer = csv.writer(f) # We used this link https://stackoverflow.com/questions/3345336/save-results-to-csv-file-with-python
        writer.writerow(['building_id'] + stat_keys)
        for bid in building_ids:
            u0, mask = load_data(LOAD_DIR, bid)
            u = jacobi_cupy_opt(u0, mask, max_iter = MAX_ITER, atol = ABS_TOL)
            stats = summary_stats(u, mask)
            writer.writerow([bid] + [stats[k] for k in stat_keys])
    
    print('\n Done.... Saved to results.csv')
    
        
            