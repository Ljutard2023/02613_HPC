"""
=============================================================================
PARTIE 4  –  GPU : CUDA kernel (Numba) + CuPy + Analyse finale
=============================================================================
Tâches couvertes : 8 (CUDA kernel Numba), 9 (CuPy), 10 (profiling nsys),
                   12 (analyse finale sur tous les bâtiments)

Expert imaginaire : Damien – Ingénieur GPU / CUDA / Architecture paralèlle
=============================================================================

CONCEPTS CLÉS GPU POUR DÉBUTANT :
  ┌─────────────────────────────────────────────────────────────────────┐
  │  CPU (Central Processing Unit)    │ GPU (Graphics Processing Unit)  │
  │  ─────────────────────────────    │ ────────────────────────────     │
  │  ~8-32 cœurs puissants            │ ~1000-10000 cœurs simples        │
  │  Bon pour tâches séquentielles    │ Excellent pour traitement massif │
  │  Cache L1/L2/L3 grands            │ Mémoire VRAM séparée de la RAM  │
  └─────────────────────────────────────────────────────────────────────┘

  Jacobi est PARFAITEMENT adapté au GPU :
  - Chaque point de la grille peut être mis à jour INDÉPENDAMMENT des autres
  - On peut lancer 514×514 = 264 000 threads simultanément !

  ORGANISATION DES THREADS CUDA :
  - Threads organisés en "blocs" (blocks), et les blocs en "grille" (grid)
  - Ex : bloc de 16×16 threads → 256 threads par bloc
  - La grille couvre toute la matrice 514×514
  - threadIdx.x, threadIdx.y  → position dans le bloc
  - blockIdx.x, blockIdx.y    → position du bloc dans la grille
  - Position globale : i = blockIdx.y * blockDim.y + threadIdx.y

  CuPy : bibliothèque NumPy-compatible qui s'exécute sur GPU.
  Il suffit de remplacer 'numpy' par 'cupy' → le calcul se fait sur le GPU !

Usage :
    python part4_gpu.py <mode> <N>
    Modes : cuda    → CUDA kernel Numba (tâche 8)
            cupy    → CuPy (tâche 9)
            analyse → Analyse finale de tous les bâtiments (tâche 12)
            all     → tout faire
    Ex : python part4_gpu.py all 10
"""

from os.path import join
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from simulate import load_data, jacobi as jacobi_reference, summary_stats

# ─────────────────────────────────────────────────────────────────────────────
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571


# =============================================================================
# TÂCHE 8  –  CUDA KERNEL avec Numba
# =============================================================================
def run_cuda_section(building_ids, n_test):
    """
    Implémente et benchmark le kernel CUDA Numba.
    Cette fonction importe numba.cuda uniquement si disponible.
    """
    try:
        from numba import cuda
    except ImportError:
        print("  [SKIP] numba.cuda non disponible.")
        return None

    if not cuda.is_available():
        print("  [SKIP] Aucun GPU CUDA disponible.")
        return None

    # ──────────────────────────────────────────────────────────────────────
    # KERNEL CUDA : s'exécute sur le GPU, lancé pour chaque point de grille
    # ──────────────────────────────────────────────────────────────────────
    @cuda.jit
    def jacobi_kernel(u, u_new, interior_mask):
        """
        Kernel CUDA : chaque thread gère UN point de la grille (i, j).

        cuda.grid(2) renvoie les indices globaux (i, j) de CE thread.
        Chaque thread :
          1. Vérifie qu'il est dans les bornes de la grille
          2. Vérifie si le point est intérieur (masque = True)
          3. Calcule la moyenne des 4 voisins → stocke dans u_new

        POURQUOI on fait UNE SEULE itération par appel de kernel :
        → Les threads d'un même bloc peuvent se synchroniser (cuda.syncthreads),
          mais PAS les blocs entre eux. Pour synchroniser TOUTE la grille entre
          chaque itération Jacobi, on doit relancer le kernel.
        """
        # Indices globaux du thread dans la grille 2D
        i, j = cuda.grid(2)

        rows = interior_mask.shape[0]  # 512
        cols = interior_mask.shape[1]  # 512

        # Vérification des bornes (la grille de threads peut dépasser la matrice)
        if 1 <= i <= rows and 1 <= j <= cols:
            # Indices dans le masque (décalage -1 car masque est 512×512)
            if interior_mask[i-1, j-1]:
                # Point intérieur : calcul de la moyenne des voisins
                u_new[i, j] = 0.25 * (
                    u[i,   j-1]   # gauche
                  + u[i,   j+1]   # droite
                  + u[i-1, j  ]   # haut
                  + u[i+1, j  ]   # bas
                )
            else:
                # Point fixe (mur ou extérieur) : on recopie sans modifier
                u_new[i, j] = u[i, j]

    def jacobi_cuda(u, interior_mask, max_iter):
        """
        Fonction helper qui encapsule le kernel CUDA.
        Prend les mêmes inputs que la référence (sauf atol : pas de critère
        d'arrêt anticipé car synchronisation globale impossible entre blocs).

        ÉTAPES :
          1. Copier les données CPU → mémoire GPU (device)
          2. Lancer le kernel N fois (ping-pong entre u_d et u_new_d)
          3. Copier le résultat GPU → CPU
        """
        # ── Configuration des blocs et de la grille ──────────────────────
        # Bloc de 16×16 threads = 256 threads/bloc (multiple de 32 = warp size)
        threads_per_block = (16, 16)
        # Nombre de blocs nécessaires pour couvrir toute la matrice 514×514
        blocks_x = math.ceil(u.shape[0] / threads_per_block[0])
        blocks_y = math.ceil(u.shape[1] / threads_per_block[1])
        blocks_per_grid = (blocks_x, blocks_y)

        # ── Transfert CPU → GPU ───────────────────────────────────────────
        u_d     = cuda.to_device(u.astype(np.float64))
        u_new_d = cuda.device_array_like(u_d)
        mask_d  = cuda.to_device(interior_mask)

        # ── Boucle d'itérations (1 kernel call = 1 itération) ─────────────
        for _ in range(max_iter):
            # Lance le kernel sur le GPU
            jacobi_kernel[blocks_per_grid, threads_per_block](u_d, u_new_d, mask_d)
            # Ping-pong : u_new devient u pour l'itération suivante
            u_d, u_new_d = u_new_d, u_d

        # ── Transfert GPU → CPU ───────────────────────────────────────────
        return u_d.copy_to_host()

    # ── Benchmark ────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════╗")
    print("║  TÂCHE 8 – CUDA Kernel       ║")
    print("╚══════════════════════════════╝")

    # Warm-up GPU (compilation + warm-up CUDA context)
    print("  Warm-up GPU...", end=' ', flush=True)
    u0_s, mask_s = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cuda(u0_s, mask_s, max_iter=1)
    print("OK")

    # Vérification de la correction
    print("  Vérification de correction...")
    for bid in building_ids[:2]:
        u0, mask = load_data(LOAD_DIR, bid)
        u_ref  = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
        u_cuda = jacobi_cuda(u0.copy(), mask, MAX_ITER)
        diff   = np.abs(u_ref - u_cuda).max()
        print(f"    Bâtiment {bid} : diff max = {diff:.2e}  "
              f"{'✓' if diff < 0.5 else '✗'}")
        # Note : sans critère d'arrêt anticipé, la convergence peut différer légèrement

    # Benchmark sur n_test bâtiments
    t_ref_total = 0.0
    t_cuda_total = 0.0
    for bid in building_ids[:n_test]:
        u0, mask = load_data(LOAD_DIR, bid)

        t0 = time.perf_counter()
        _ = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
        t_ref_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = jacobi_cuda(u0.copy(), mask, MAX_ITER)
        t_cuda_total += time.perf_counter() - t0

    print(f"\n  Référence NumPy : {t_ref_total:.2f}s ({n_test} bâtiments)")
    print(f"  CUDA Kernel     : {t_cuda_total:.2f}s ({n_test} bâtiments)")
    print(f"  Speed-up        : {t_ref_total/t_cuda_total:.1f}x")
    est = (t_cuda_total / n_test) * N_TOTAL
    print(f"  Estimation 4571 : {est/60:.1f} min")

    print("""
  NOTE sur le profiling nsys (tâche 10) :
  ────────────────────────────────────────────────────────────────
  Pour profiler avec nsys :
      nsys profile --trace=cuda,nvtx python part4_gpu.py cuda 5
      nsys-ui report1.nsys-rep    (ouvrir l'interface graphique)

  Problème typique identifié par nsys :
  → Les transferts mémoire CPU↔GPU (cudaMemcpy) dominent le temps !
    Chaque bâtiment nécessite : 514×514×8 octets × 2 = ~4 MB de transfert.
    Si on traite 10 bâtiments → 40 MB de transfert pour chaque sens.

  SOLUTION (tâche 10 fix) :
  → Pré-charger TOUS les bâtiments sur le GPU avant de démarrer la boucle.
  → Utiliser des streams CUDA pour chevaucher calcul et transfert mémoire.
  ────────────────────────────────────────────────────────────────
""")

    return t_cuda_total / n_test


# =============================================================================
# TÂCHE 9  –  CuPy (version GPU du code NumPy)
# =============================================================================
def run_cupy_section(building_ids, n_test):
    """
    Implémente et benchmark la version CuPy.

    CuPy = NumPy pour GPU.
    import cupy as cp
    cp.array(data)   → copie data sur le GPU
    cp.asnumpy(arr)  → copie arr vers le CPU
    Toutes les opérations NumPy fonctionnent TELLES QUELLES sur le GPU !
    """
    try:
        import cupy as cp
    except ImportError:
        print("  [SKIP] CuPy non disponible. Installe : pip install cupy-cuda12x")
        return None

    print("\n╔══════════════════════════════╗")
    print("║  TÂCHE 9 – CuPy              ║")
    print("╚══════════════════════════════╝")

    def jacobi_cupy(u, interior_mask, max_iter, atol=1e-4):
        """
        Version CuPy de Jacobi.

        Strictement identique au code NumPy de référence,
        sauf qu'on travaille avec des tableaux cp (GPU) au lieu de np (CPU).

        POURQUOI C'EST RAPIDE :
        → Toutes les opérations (+, *, indexation) sont exécutées sur le GPU
          avec des milliers de threads en parallèle.
        → float(delta) force un transfert GPU→CPU minimal juste pour le
          critère d'arrêt. C'est le seul "aller-retour" GPU↔CPU par itération.
        """
        # Copier les données CPU → GPU
        u_gpu    = cp.array(u, dtype=cp.float64)
        mask_gpu = cp.array(interior_mask)

        for i in range(max_iter):
            # Calcul de la moyenne des voisins (IDENTIQUE à NumPy !)
            u_new = 0.25 * (
                u_gpu[1:-1, :-2]   # voisin gauche
              + u_gpu[1:-1, 2:]    # voisin droite
              + u_gpu[:-2, 1:-1]   # voisin haut
              + u_gpu[2:,  1:-1]   # voisin bas
            )
            # Calcul de la variation maximale pour le critère d'arrêt
            delta = cp.abs(u_gpu[1:-1, 1:-1][mask_gpu] - u_new[mask_gpu]).max()
            # Mise à jour uniquement des points intérieurs
            u_gpu[1:-1, 1:-1][mask_gpu] = u_new[mask_gpu]

            # float() force un transfert GPU→CPU pour comparer avec atol
            if float(delta) < atol:
                break

        # Copier le résultat GPU → CPU
        return cp.asnumpy(u_gpu)

    # ── Warm-up ──────────────────────────────────────────────────────────
    print("  Warm-up GPU...", end=' ', flush=True)
    u0_s, mask_s = load_data(LOAD_DIR, building_ids[0])
    _ = jacobi_cupy(u0_s, mask_s, max_iter=1)
    print("OK")

    # ── Vérification ─────────────────────────────────────────────────────
    print("  Vérification de correction...")
    for bid in building_ids[:2]:
        u0, mask = load_data(LOAD_DIR, bid)
        u_ref  = jacobi_reference(u0,       mask, MAX_ITER, ABS_TOL)
        u_cupy = jacobi_cupy(u0.copy(), mask, MAX_ITER, ABS_TOL)
        diff   = np.abs(u_ref - u_cupy).max()
        print(f"    Bâtiment {bid} : diff max = {diff:.2e}  {'✓' if diff < 1e-3 else '✗'}")

    # ── Benchmark ─────────────────────────────────────────────────────────
    t_ref   = 0.0
    t_cupy  = 0.0
    for bid in building_ids[:n_test]:
        u0, mask = load_data(LOAD_DIR, bid)

        t0 = time.perf_counter()
        _ = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
        t_ref += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = jacobi_cupy(u0.copy(), mask, MAX_ITER, ABS_TOL)
        t_cupy += time.perf_counter() - t0

    print(f"\n  Référence NumPy : {t_ref:.2f}s ({n_test} bâtiments)")
    print(f"  CuPy            : {t_cupy:.2f}s ({n_test} bâtiments)")
    print(f"  Speed-up        : {t_ref/t_cupy:.1f}x")
    est = (t_cupy / n_test) * N_TOTAL
    print(f"  Estimation 4571 : {est/60:.1f} min")

    print("""
  OBSERVATION SURPRENANTE possible :
  → CuPy peut parfois être PLUS LENT que NumPy sur peu de bâtiments !
  Raison : le transfert mémoire CPU↔GPU a un coût fixe important.
  Pour de petites grilles ou peu d'itérations, ce surcoût domine.
  → CuPy est avantageux quand la grille est grande ET qu'il y a
    beaucoup d'itérations (le coût de transfert est amorti).
  → Fix tâche 10 : pré-charger toutes les données sur le GPU
    avant de démarrer et récupérer tous les résultats à la fin
    → minimise les transferts = gain majeur.
""")
    return t_cupy / n_test


# =============================================================================
# TÂCHE 9 FIX / TÂCHE 10  –  CuPy avec pré-chargement GPU
# =============================================================================
def run_cupy_optimized(building_ids, n_test):
    """
    Version CuPy optimisée : toutes les données sont chargées sur le GPU
    UNE SEULE FOIS avant la boucle → minimise les transferts mémoire.
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    print("\n╔══════════════════════════════╗")
    print("║  TÂCHE 10 – CuPy optimisé   ║")
    print("╚══════════════════════════════╝")

    ids = building_ids[:n_test]

    # ── Charger TOUT sur le GPU d'un coup ─────────────────────────────────
    print("  Chargement des données sur GPU...", end=' ', flush=True)
    t0 = time.perf_counter()
    all_u_gpu    = []
    all_mask_gpu = []
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        all_u_gpu.append(cp.array(u0, dtype=cp.float64))
        all_mask_gpu.append(cp.array(mask))
    t_transfer = time.perf_counter() - t0
    print(f"OK ({t_transfer:.2f}s)")

    # ── Simulation sur GPU ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    for u_gpu, mask_gpu in zip(all_u_gpu, all_mask_gpu):
        for _ in range(MAX_ITER):
            u_new = 0.25 * (
                u_gpu[1:-1, :-2] + u_gpu[1:-1, 2:]
              + u_gpu[:-2, 1:-1] + u_gpu[2:,  1:-1]
            )
            delta = cp.abs(u_gpu[1:-1, 1:-1][mask_gpu] - u_new[mask_gpu]).max()
            u_gpu[1:-1, 1:-1][mask_gpu] = u_new[mask_gpu]
            if float(delta) < ABS_TOL:
                break
    # Synchronisation GPU (attendre que tous les calculs soient finis)
    cp.cuda.stream.get_current_stream().synchronize()
    t_sim = time.perf_counter() - t0
    print(f"  Simulation GPU  : {t_sim:.2f}s")

    # ── Récupérer les résultats d'un coup ─────────────────────────────────
    t0 = time.perf_counter()
    results = []
    for bid, u_gpu, mask_gpu in zip(ids, all_u_gpu, all_mask_gpu):
        u_cpu   = cp.asnumpy(u_gpu)
        mask_cpu = cp.asnumpy(mask_gpu)
        stats = summary_stats(u_cpu, mask_cpu)
        results.append((bid, stats))
    t_back = time.perf_counter() - t0

    t_total = t_transfer + t_sim + t_back
    print(f"  Transfert retour : {t_back:.2f}s")
    print(f"  TOTAL            : {t_total:.2f}s")
    est = (t_total / n_test) * N_TOTAL
    print(f"  Estimation 4571  : {est/60:.1f} min")

    return results, t_total / n_test


# =============================================================================
# TÂCHE 12  –  Analyse finale sur TOUS les bâtiments
# =============================================================================
def final_analysis(results_csv_path=None, building_ids=None, n_sample=100):
    """
    Analyse statistique complète des résultats de simulation.

    Si on n'a pas de résultats précalculés, on les calcule sur un sous-ensemble.
    En production, utiliser la méthode la plus rapide disponible pour 4571 bâtiments.

    Répond aux questions :
      a) Distribution des températures moyennes (histogrammes)
      b) Température moyenne globale
      c) Écart-type moyen de température
      d) Combien de bâtiments ont ≥50% de surface au-dessus de 18°C ?
      e) Combien de bâtiments ont ≥50% de surface en-dessous de 15°C ?
    """
    print("\n╔══════════════════════════════╗")
    print("║  TÂCHE 12 – Analyse finale   ║")
    print("╚══════════════════════════════╝")

    # ── Charger ou calculer les résultats ─────────────────────────────────
    if results_csv_path:
        df = pd.read_csv(results_csv_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        print(f"  CSV chargé : {len(df)} bâtiments")
    else:
        # Calcul sur un échantillon si pas de CSV
        print(f"  Calcul sur {n_sample} bâtiments...")
        rows = []
        for bid in building_ids[:n_sample]:
            u0, mask = load_data(LOAD_DIR, bid)
            u = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
            stats = summary_stats(u, mask)
            rows.append({
                'building_id':  bid,
                'mean_temp':    stats['mean_temp'],
                'std_temp':     stats['std_temp'],
                'pct_above_18': stats['pct_above_18'],
                'pct_below_15': stats['pct_below_15'],
            })
        df = pd.DataFrame(rows)
        df.to_csv('results_all.csv', index=False)
        print(f"  Résultats sauvegardés dans results_all.csv")

    # ── Question b) : Température moyenne globale ──────────────────────────
    mean_of_means = df['mean_temp'].mean()
    print(f"\n  (b) Température moyenne globale : {mean_of_means:.2f}°C")

    # ── Question c) : Écart-type moyen ────────────────────────────────────
    mean_std = df['std_temp'].mean()
    print(f"  (c) Écart-type moyen            : {mean_std:.2f}°C")

    # ── Question d) : Bâtiments avec ≥50% surface > 18°C ──────────────────
    n_above_18 = (df['pct_above_18'] >= 50).sum()
    print(f"  (d) Bâtiments avec ≥50% surface >18°C : "
          f"{n_above_18} / {len(df)} ({n_above_18/len(df)*100:.1f}%)")

    # ── Question e) : Bâtiments avec ≥50% surface < 15°C ──────────────────
    n_below_15 = (df['pct_below_15'] >= 50).sum()
    print(f"  (e) Bâtiments avec ≥50% surface <15°C : "
          f"{n_below_15} / {len(df)} ({n_below_15/len(df)*100:.1f}%)")

    # ── Question a) : Histogrammes ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Distribution des statistiques de température\n'
                 f'(Wall Heating – {len(df)} bâtiments)', fontsize=13)

    configs = [
        (axes[0, 0], 'mean_temp',    'Température moyenne (°C)',
         'Distribution des températures moyennes', 'royalblue',
         [18], ['≥18°C recommandé']),
        (axes[0, 1], 'std_temp',     'Écart-type (°C)',
         'Distribution des écarts-types', 'orange', [], []),
        (axes[1, 0], 'pct_above_18', '% surface > 18°C',
         '% surface au-dessus de 18°C', 'green',
         [50], ['50% seuil']),
        (axes[1, 1], 'pct_below_15', '% surface < 15°C',
         '% surface en-dessous de 15°C', 'tomato',
         [50], ['50% seuil inconfort']),
    ]

    for ax, col, xlabel, title, color, vlines, vlabels in configs:
        ax.hist(df[col], bins=30, color=color, alpha=0.75, edgecolor='white')
        for vl, vlab in zip(vlines, vlabels):
            ax.axvline(vl, color='black', linestyle='--', lw=1.5, label=vlab)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Nombre de bâtiments', fontsize=10)
        ax.set_title(title, fontsize=11)
        if vlines:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('part4_task12_analysis.png', dpi=150, bbox_inches='tight')
    print("\n  → Sauvegardé : part4_task12_analysis.png")
    plt.close()

    # ── Résumé qualitatif ─────────────────────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │             RÉSUMÉ – WALL HEATING EVALUATION                │
  ├─────────────────────────────────────────────────────────────┤
  │  Temp. moyenne des bâtiments    : {mean_of_means:>5.1f}°C                   │
  │  Écart-type moyen               : {mean_std:>5.1f}°C                   │
  │  Bâtiments 'chauds' (>18°C×50%): {n_above_18:>4d} / {len(df):<4d}             │
  │  Bâtiments 'froids' (<15°C×50%): {n_below_15:>4d} / {len(df):<4d}             │
  └─────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    mode   = sys.argv[1] if len(sys.argv) > 1 else 'all'
    N      = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print("=" * 55)
    print("  PARTIE 4 – GPU (CUDA + CuPy) + Analyse finale")
    print("=" * 55)

    if mode in ('cuda', 'all'):
        run_cuda_section(building_ids, n_test=N)

    if mode in ('cupy', 'all'):
        run_cupy_section(building_ids, n_test=N)
        run_cupy_optimized(building_ids, n_test=N)

    if mode in ('analyse', 'all'):
        # Si un CSV de résultats existe, l'utiliser ; sinon calculer
        import os
        csv_path = 'results_all.csv' if os.path.exists('results_all.csv') else None
        final_analysis(
            results_csv_path=csv_path,
            building_ids=building_ids,
            n_sample=min(N, len(building_ids))
        )
