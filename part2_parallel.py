"""
=============================================================================
PARTIE 2  –  Parallélisation CPU (ordonnancement statique & dynamique)
=============================================================================
Tâches couvertes : 5 (parallélisation statique + plots speed-up + Amdahl)
                   6 (ordonnancement dynamique)

Expert imaginaire : Bob – Spécialiste parallélisme CPU / OpenMP / MPI
=============================================================================

CONCEPT CLÉ POUR DÉBUTANT :
  Par défaut, Python traite UN bâtiment à la fois (séquentiel).
  Si on a 8 cœurs CPU, on peut traiter 8 bâtiments EN MÊME TEMPS → 8× plus vite.

  - Ordonnancement STATIQUE  : on divise d'avance le travail en parts égales.
    Ex : 100 bâtiments, 4 workers → chacun traite exactement 25 bâtiments.
    Problème : si certains bâtiments sont plus longs que d'autres, certains
    workers finissent tôt et attendent les autres.

  - Ordonnancement DYNAMIQUE : chaque worker prend un bâtiment à la fois
    dans une file d'attente. Quand il a fini, il en prend un autre.
    Plus flexible → meilleur équilibrage de charge (load balancing).

  Loi d'Amdahl : si une fraction p du code est parallélisable, le gain
  maximum avec N cœurs est : S(N) = 1 / ((1-p) + p/N)
  → même avec ∞ cœurs, on ne peut jamais dépasser 1/(1-p).

Usage :
    python part2_parallel.py <N> <mode>
    Ex :  python part2_parallel.py 100 static
          python part2_parallel.py 100 dynamic
          python part2_parallel.py 100 plot     ← génère les graphes de speed-up
"""

from os.path import join
import sys
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from simulate import load_data, jacobi, summary_stats

# ─────────────────────────────────────────────────────────────────────────────
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571   # nombre total de bâtiments


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION WORKER  –  traite UN seul bâtiment
# ─────────────────────────────────────────────────────────────────────────────
def process_building(bid):
    """
    Charge les données d'un bâtiment, lance Jacobi et retourne les stats.
    Cette fonction est appelée en parallèle dans chaque worker.
    Elle doit être définie au niveau module (pas dans une autre fonction)
    pour que multiprocessing puisse la sérialiser (pickle).
    """
    u0, mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, mask)
    return bid, stats


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 5a  –  Parallélisation STATIQUE
# ─────────────────────────────────────────────────────────────────────────────
def run_static(building_ids, n_workers):
    """
    Utilise multiprocessing.Pool.map() pour la distribution statique.

    Pool.map(f, liste) :
      - Divise 'liste' en chunks de taille ≈ len(liste)/n_workers
      - Envoie chaque chunk à un worker
      - Attend que TOUS les workers aient fini
      → Distribution STATIQUE : le découpage est fait UNE SEULE FOIS au début.
    """
    with mp.Pool(processes=n_workers) as pool:
        # pool.map divise automatiquement la liste en parts égales
        results = pool.map(process_building, building_ids)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 6  –  Parallélisation DYNAMIQUE
# ─────────────────────────────────────────────────────────────────────────────
def run_dynamic(building_ids, n_workers):
    """
    Utilise multiprocessing.Pool.imap_unordered() pour la distribution dynamique.

    Pool.imap_unordered(f, liste, chunksize=1) :
      - chunksize=1 → chaque worker prend UN bâtiment à la fois
      - Dès qu'un worker libère, il prend le prochain bâtiment en attente
      → Distribution DYNAMIQUE : aucun worker ne reste inactif longtemps.

    'imap' = itérateur map (lazy) : retourne les résultats au fur et à mesure.
    'unordered' = l'ordre de retour n'est pas garanti (mais c'est plus rapide).
    """
    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(process_building, building_ids,
                                           chunksize=1))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 5a  –  Mesurer le speed-up en fonction du nombre de workers
# ─────────────────────────────────────────────────────────────────────────────
def measure_speedup(building_ids, max_workers=None, mode='static'):
    """
    Mesure le temps pour différents nombres de workers et calcule le speed-up.

    Speed-up S(N) = T(1) / T(N)
      - T(1) = temps avec 1 worker (référence)
      - T(N) = temps avec N workers
      - Si S(N) = N → speed-up parfait (linéaire), jamais atteint en pratique.
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    worker_counts = [1, 2, 4, 8, 16, 32]
    worker_counts = [w for w in worker_counts if w <= max_workers]

    times   = []
    speedups = []

    print(f"\n{'Workers':>8} | {'Temps (s)':>10} | {'Speed-up':>10}")
    print("-" * 35)

    t_ref = None  # temps avec 1 worker = référence

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


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 5b,c  –  Loi d'Amdahl
# ─────────────────────────────────────────────────────────────────────────────
def amdahl_analysis(worker_counts, speedups):
    """
    Estime la fraction parallèle p selon la loi d'Amdahl.

    Loi d'Amdahl : S(N) = 1 / ((1-p) + p/N)
    On résout pour p à partir du meilleur speed-up mesuré.

    Avec le speed-up S messuré à N workers :
        p = (1/S - 1) / (1/N - 1)
    """
    print("\n─── Analyse Amdahl ────────────────────────────────────────")

    # Utilise le speed-up avec le plus grand N pour estimer p
    best_idx = -1   # dernier point = plus grand N
    N_best   = worker_counts[best_idx]
    S_best   = speedups[best_idx]

    if N_best > 1:
        p = (1/S_best - 1) / (1/N_best - 1)
        p = max(0.0, min(1.0, p))  # borne entre 0 et 1
    else:
        p = 0.0

    print(f"  Fraction parallèle estimée : p ≈ {p*100:.1f}%")
    print(f"  Fraction séquentielle      : 1-p ≈ {(1-p)*100:.1f}%")

    S_max = 1 / (1 - p) if p < 1 else float('inf')
    print(f"  Speed-up théorique maximum (∞ cœurs) : {S_max:.1f}x")
    print(f"  Speed-up atteint avec {N_best} workers : {S_best:.2f}x")
    print(f"  Efficacité avec {N_best} workers : {S_best/N_best*100:.0f}%")

    return p, S_max


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 5a  –  Générer les graphes de speed-up
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedups(wc_static, sp_static, wc_dynamic, sp_dynamic,
                  p_static, p_dynamic, n_buildings):
    """
    Trace le speed-up mesuré vs. speed-up idéal vs. speed-up Amdahl.
    """
    # Courbes Amdahl théoriques
    N_range = np.linspace(1, max(max(wc_static), max(wc_dynamic)), 200)

    def amdahl_curve(p, n_arr):
        return 1 / ((1 - p) + p / n_arr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Speed-up — {n_buildings} bâtiments', fontsize=13)

    for ax, label, wc, sp, p in [
        (axes[0], 'Statique',  wc_static,  sp_static,  p_static),
        (axes[1], 'Dynamique', wc_dynamic, sp_dynamic, p_dynamic),
    ]:
        ax.plot(wc, sp,      'o-', color='royalblue',  lw=2,
                label='Speed-up mesuré')
        ax.plot(N_range, N_range, '--', color='gray', lw=1.5,
                label='Idéal (linéaire)')
        ax.plot(N_range, amdahl_curve(p, N_range), ':', color='tomato', lw=2,
                label=f'Amdahl (p={p*100:.0f}%)')
        ax.set_xlabel('Nombre de workers (cœurs CPU)')
        ax.set_ylabel('Speed-up S(N)')
        ax.set_title(f'Ordonnancement {label}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=1)

    plt.tight_layout()
    plt.savefig('part2_speedup.png', dpi=150, bbox_inches='tight')
    print("\n  → Sauvegardé : part2_speedup.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Lecture des arguments
    N      = int(sys.argv[1])  if len(sys.argv) > 1 else 20
    mode   = sys.argv[2]       if len(sys.argv) > 2 else 'all'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    max_w = mp.cpu_count()
    print(f"CPU disponibles : {max_w}")
    print(f"Bâtiments utilisés pour les tests : {N}")

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    if mode in ('static', 'all'):
        # ── STATIQUE ──────────────────────────────────────────────────────
        print("\n╔══════════════════════════════╗")
        print("║  ORDONNANCEMENT STATIQUE     ║")
        print("╚══════════════════════════════╝")
        wc_s, t_s, sp_s = measure_speedup(building_ids, max_w, mode='static')
        p_s, smax_s     = amdahl_analysis(wc_s, sp_s)

        # Estimation temps total avec le meilleur setting statique
        best_t_s = min(t_s)
        best_w_s = wc_s[t_s.index(best_t_s)]
        estimated_s = (best_t_s / N) * N_TOTAL
        print(f"\n  Meilleur temps statique : {best_t_s:.2f}s ({N} bâtiments, "
              f"{best_w_s} workers)")
        print(f"  → Estimation pour {N_TOTAL} bâtiments : "
              f"{estimated_s/60:.1f} min ({estimated_s/3600:.2f}h)")

    if mode in ('dynamic', 'all'):
        # ── DYNAMIQUE ─────────────────────────────────────────────────────
        print("\n╔══════════════════════════════╗")
        print("║  ORDONNANCEMENT DYNAMIQUE    ║")
        print("╚══════════════════════════════╝")
        wc_d, t_d, sp_d = measure_speedup(building_ids, max_w, mode='dynamic')
        p_d, smax_d     = amdahl_analysis(wc_d, sp_d)

        best_t_d = min(t_d)
        best_w_d = wc_d[t_d.index(best_t_d)]
        estimated_d = (best_t_d / N) * N_TOTAL
        print(f"\n  Meilleur temps dynamique : {best_t_d:.2f}s ({N} bâtiments, "
              f"{best_w_d} workers)")
        print(f"  → Estimation pour {N_TOTAL} bâtiments : "
              f"{estimated_d/60:.1f} min ({estimated_d/3600:.2f}h)")

    if mode == 'all':
        # ── COMPARAISON & GRAPHES ─────────────────────────────────────────
        print("\n╔══════════════════════════════╗")
        print("║  GRAPHES DE SPEED-UP         ║")
        print("╚══════════════════════════════╝")
        plot_speedups(wc_s, sp_s, wc_d, sp_d, p_s, p_d, N)

        print("\n╔══════════════════════════════╗")
        print("║  RÉSULTATS CSV (dynamique)   ║")
        print("╚══════════════════════════════╝")
        results = run_dynamic(building_ids, best_w_d)
        print('building_id, ' + ', '.join(stat_keys))
        for bid, stats in results:
            print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
