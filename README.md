# Wall Heating — 02613 HPC Mini-Project

Simulation of steady-state heat distribution in 4,571 Swiss building floorplans,
using the Jacobi iterative method to solve Laplace's equation.
The goal is to optimise the reference implementation as much as possible.

---

## File structure
Project/
├── simulate.py              # Reference implementation (provided by course)
├── jacobi_profile.py        # Line-by-line profiling script (Task 4)
├── part1_explore.py         # Task 1-3: data exploration and visualisation
├── part2_parallel.py        # Tasks 5-6: CPU parallelisation (static & dynamic)
├── part3_numba.py           # Task 7: Numba JIT optimisation
├── make_histogram.py        # Task 12: final statistics and histograms
├── job_part1.sh             # LSF job: Part 1
├── job_part1_kernprof.sh    # LSF job: kernprof profiling
├── job_part2.sh             # LSF job: Part 2 (16 cores)
├── job_part3.sh             # LSF job: Part 3 (8 cores)
├── job_part4_cupy.sh        # LSF job: CuPy naive benchmark (GPU)
├── job_cupy_optimized.sh    # LSF job: CuPy optimised benchmark (GPU)
├── results_all.csv          # Final CSV results for all 4,571 buildings
└── logs/                    # LSF job output logs
---

## Installation

On the DTU HPC cluster:
```bash
pip install numpy matplotlib pandas numba cupy-cuda12x line_profiler
```

---

## Usage

### Submit jobs on DTU HPC (LSF)
```bash
bsub < job_part1.sh             # Tasks 1-3: exploration, timing, visualisation
bsub < job_part1_kernprof.sh    # Task 4: kernprof profiling
bsub < job_part2.sh             # Tasks 5-6: CPU parallelisation speed-up
bsub < job_part3.sh             # Task 7: Numba JIT benchmark
bsub < job_part4_cupy.sh        # Task 9: CuPy naive GPU benchmark
bsub < job_cupy_optimized.sh    # Task 10: CuPy optimised GPU benchmark
```

### Run locally (no GPU)
```bash
python part1_explore.py
python part2_parallel.py 10 all
python part3_numba.py 5
python make_histogram.py        # requires results_all.csv
```

---

## Results summary

| Method                    | Speed-up | Estimated time (4,571 buildings) |
|--------------------------|----------|----------------------------------|
| Reference NumPy           | 1×       | 14.2 h                           |
| Static CPU (16 workers)   | 4.7×     | 3.4 h                            |
| Dynamic CPU (16 workers)  | 4.7×     | 3.4 h                            |
| Numba JIT sequential      | 3.6×     | 3.0 h                            |
| Numba JIT parallel (8T)   | 51×      | ~13 min                          |
| CuPy naive (V100)         | 3.2×     | 3.4 h                            |
| CuPy optimised (V100)     | 2351×    | ~18 s                            |

---

## Background

The Jacobi method updates each interior grid point as the average of its 4 neighbours:
u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4

Repeated until convergence (max change < atol) or max_iter reached.
Grid size: 514×514 = 264,196 points × up to 20,000 iterations per building.

---

## Authors

Ibrahim Kerpic (s224403), Sami Rana (s224386), Lucas Jutard (s253050)
DTU — 02613 Python and High Performance Computing, April 2026
