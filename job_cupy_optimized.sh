#!/bin/bash
#BSUB -J cupy_opt
#BSUB -o logs/cupy_opt_%J.out
#BSUB -e logs/cupy_opt_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 00:30
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/12.1
source /zhome/3f/9/223204/projects/.venv/bin/activate

python - << 'PYEOF'
import time, cupy as cp
from os.path import join
from simulate import load_data, jacobi as jacobi_ref

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER, ABS_TOL = 20_000, 1e-4

with open(join(LOAD_DIR, 'building_ids.txt')) as f:
    building_ids = f.read().splitlines()[:10]

def jacobi_cupy_naive(u, mask, max_iter, atol=1e-4):
    """Naive: float(delta) forces a GPU->CPU memory transfer every iteration."""
    u_gpu, mask_gpu = cp.array(u, dtype=cp.float64), cp.array(mask)
    for i in range(max_iter):
        u_new = 0.25*(u_gpu[1:-1,:-2]+u_gpu[1:-1,2:]+u_gpu[:-2,1:-1]+u_gpu[2:,1:-1])
        delta = cp.abs(u_gpu[1:-1,1:-1][mask_gpu] - u_new[mask_gpu]).max()
        u_gpu[1:-1,1:-1][mask_gpu] = u_new[mask_gpu]
        if float(delta) < atol:
            break
    return cp.asnumpy(u_gpu)

def jacobi_cupy_opt(u, mask, max_iter, check_every=50):
    """Optimised: check convergence every 50 steps to minimise GPU->CPU transfers."""
    u_gpu, mask_gpu = cp.array(u, dtype=cp.float64), cp.array(mask)
    for i in range(max_iter):
        u_new = 0.25*(u_gpu[1:-1,:-2]+u_gpu[1:-1,2:]+u_gpu[:-2,1:-1]+u_gpu[2:,1:-1])
        u_gpu[1:-1,1:-1][mask_gpu] = u_new[mask_gpu]
        if i % check_every == 0:
            delta = cp.abs(u_gpu[1:-1,1:-1][mask_gpu] - u_new[mask_gpu]).max()
            if float(delta) < ABS_TOL:
                break
    return cp.asnumpy(u_gpu)

# Warm-up
u0, mask = load_data(LOAD_DIR, building_ids[0])
_ = jacobi_cupy_naive(u0, mask, 1)
_ = jacobi_cupy_opt(u0, mask, 1)
print("Warm-up OK")

t0 = time.perf_counter()
for bid in building_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi_ref(u0, mask, MAX_ITER, ABS_TOL)
t_ref = time.perf_counter() - t0

t0 = time.perf_counter()
for bid in building_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi_cupy_naive(u0, mask, MAX_ITER, ABS_TOL)
t_naive = time.perf_counter() - t0

t0 = time.perf_counter()
for bid in building_ids:
    u0, mask = load_data(LOAD_DIR, bid)
    jacobi_cupy_opt(u0, mask, MAX_ITER)
t_opt = time.perf_counter() - t0

print(f"Reference NumPy  : {t_ref:.2f}s  ->  1.0x")
print(f"CuPy naive       : {t_naive:.2f}s  ->  {t_ref/t_naive:.1f}x")
print(f"CuPy optimised   : {t_opt:.2f}s  ->  {t_ref/t_opt:.1f}x")
print(f"Optimisation gain: {t_naive/t_opt:.1f}x")
print(f"Estimated 4571   : {(t_opt/10)*4571/60:.1f} min")
PYEOF
