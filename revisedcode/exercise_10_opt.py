""" 
task 10
"""
import time
import cupy as cp
from os.path import join
from simulate import load_data, jacobi as jacobi_ref

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER, ABS_TOL = 20_000, 1e-4

with open(join(LOAD_DIR, 'building_ids.txt')) as f:
    building_ids = f.read().splitlines()[:10]

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

# Warm-up
u0, mask = load_data(LOAD_DIR, building_ids[0])
_ = jacobi_cupy_opt(u0, mask, max_iter=1)
print("Warm-up OK", flush=True)

t0 = time.perf_counter()
for bid in building_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi_ref(u0, mask, MAX_ITER, ABS_TOL)
t_ref = time.perf_counter() - t0

t0 = time.perf_counter()
for bid in building_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi_cupy_opt(u0, mask, max_iter = MAX_ITER, atol = ABS_TOL)
t_cupy = time.perf_counter() - t0

print(f"Reference NumPy : {t_ref:.2f}s")
print(f"CuPy GPU        : {t_cupy:.2f}s")
print(f"Speed-up        : {t_ref/t_cupy:.1f}x")
print(f"Estimated 4571  : {(t_cupy/10)*4571/60:.1f} min")