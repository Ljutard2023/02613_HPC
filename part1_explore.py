"""
=============================================================================
PARTIE 1  –  Exploration des données et analyse de base
=============================================================================
Tâches couvertes : 1 (visualisation), 2 (timing), 3 (visualiser résultats),
                   4 (profiling avec kernprof)

Expert imaginaire : Alice – Scientifique en données / Visualisation HPC
=============================================================================

COMMENT LIRE CE FICHIER :
  Chaque section est clairement délimitée. Les commentaires expliquent chaque
  concept pour un débutant. Lance le script avec :
      python part1_explore.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS  –  on charge les bibliothèques nécessaires
# ─────────────────────────────────────────────────────────────────────────────
from os.path import join
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# On réutilise les fonctions du script de référence
from simulate import load_data, jacobi, summary_stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 1 : Visualiser les données d'entrée pour quelques bâtiments
# ─────────────────────────────────────────────────────────────────────────────
def task1_visualize_input(building_ids, n=4):
    """
    Affiche la grille initiale (domaine) ET le masque intérieur pour n bâtiments.

    La grille 'domain' :
      - Valeur 25  → mur intérieur chauffant (chaud)
      - Valeur  5  → mur porteur froid
      - Valeur  0  → air dans les pièces (à calculer)

    Le masque 'interior' :
      - 1 (blanc) → points à l'intérieur des pièces → seront mis à jour
      - 0 (noir)  → murs ou extérieur → fixes
    """
    print("=" * 60)
    print("TÂCHE 1 – Visualisation des données d'entrée")
    print("=" * 60)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Données d'entrée : domaine et masques intérieurs", fontsize=14)

    for idx in range(n):
        bid = building_ids[idx]
        u0, mask = load_data(LOAD_DIR, bid)

        # Ligne du haut : grille de température initiale
        im = axes[0, idx].imshow(u0, cmap='hot', vmin=0, vmax=25)
        axes[0, idx].set_title(f'Domaine\nID: {bid}', fontsize=9)
        axes[0, idx].axis('off')

        # Ligne du bas : masque des points intérieurs
        axes[1, idx].imshow(mask, cmap='gray')
        axes[1, idx].set_title(f'Masque intérieur\nID: {bid}', fontsize=9)
        axes[1, idx].axis('off')

    plt.colorbar(im, ax=axes[0, :], label='Température (°C)', shrink=0.6)
    plt.tight_layout()
    plt.savefig('part1_task1_input_data.png', dpi=150, bbox_inches='tight')
    print("  → Sauvegardé : part1_task1_input_data.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 2 : Mesurer le temps d'exécution sur N bâtiments
# ─────────────────────────────────────────────────────────────────────────────
def task2_timing(building_ids, n_test=10):
    """
    Mesure le temps pour traiter n_test bâtiments, puis extrapole à 4571.

    Pourquoi extrapoler ?
      On ne peut pas attendre des heures sur un laptop, mais on peut
      mesurer sur 10 bâtiments et multiplier pour estimer le temps total.
    """
    print("\n" + "=" * 60)
    print(f"TÂCHE 2 – Timing sur {n_test} bâtiments")
    print("=" * 60)

    ids_test = building_ids[:n_test]

    # ── Chargement des données ──────────────────────────────────────────────
    t_start_load = time.perf_counter()
    all_u0    = []
    all_masks = []
    for bid in ids_test:
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)
    t_load = time.perf_counter() - t_start_load

    # ── Simulation Jacobi ───────────────────────────────────────────────────
    t_start_sim = time.perf_counter()
    for u0, mask in zip(all_u0, all_masks):
        jacobi(u0, mask, MAX_ITER, ABS_TOL)
    t_sim = time.perf_counter() - t_start_sim

    t_total = t_load + t_sim
    n_total = 4571  # nombre total de bâtiments dans le dataset

    print(f"  Chargement    : {t_load:.2f}s pour {n_test} bâtiments")
    print(f"  Simulation    : {t_sim:.2f}s pour {n_test} bâtiments")
    print(f"  Total         : {t_total:.2f}s pour {n_test} bâtiments")
    print(f"  Par bâtiment  : {t_total/n_test:.2f}s")
    t_extrap = (t_total / n_test) * n_total
    print(f"\n  ➜ Estimation pour {n_total} bâtiments : {t_extrap/60:.1f} minutes "
          f"({t_extrap/3600:.2f} heures)")
    print("  ➜ C'est BEAUCOUP trop lent → besoin d'optimisation !")

    return t_total / n_test   # secondes par bâtiment


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 3 : Visualiser les résultats de simulation
# ─────────────────────────────────────────────────────────────────────────────
def task3_visualize_results(building_ids, n=4):
    """
    Lance la simulation et affiche la carte de chaleur résultante.

    Après convergence de Jacobi, chaque pixel des pièces a une température
    calculée. On affiche cette carte avec une colormap 'inferno' (noir→jaune).
    """
    print("\n" + "=" * 60)
    print("TÂCHE 3 – Visualisation des résultats de simulation")
    print("=" * 60)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Résultats de simulation : distribution de chaleur", fontsize=14)

    for idx in range(n):
        bid = building_ids[idx]
        u0, mask = load_data(LOAD_DIR, bid)

        print(f"  Simulation bâtiment {bid} ...", end=' ', flush=True)
        t0 = time.perf_counter()
        u_result = jacobi(u0, mask, MAX_ITER, ABS_TOL)
        dt = time.perf_counter() - t0
        print(f"{dt:.1f}s")

        # Masquer l'extérieur (là où mask=0 ET u=0) pour une belle visualisation
        display = u_result.copy()
        exterior = (display == 0)  # pixels extérieurs = 0 après simulation
        display[exterior] = np.nan  # NaN → transparent sur le graphe

        im = axes[idx].imshow(display, cmap='inferno', vmin=5, vmax=25)
        axes[idx].set_title(f'ID: {bid}', fontsize=9)
        axes[idx].axis('off')

        # Statistiques rapides
        stats = summary_stats(u_result, mask)
        info = (f"moy={stats['mean_temp']:.1f}°C\n"
                f">18°C: {stats['pct_above_18']:.0f}%")
        axes[idx].text(0.02, 0.02, info, transform=axes[idx].transAxes,
                       fontsize=7, color='white',
                       bbox=dict(facecolor='black', alpha=0.5))

    plt.colorbar(im, ax=axes, label='Température (°C)', shrink=0.8)
    plt.tight_layout()
    plt.savefig('part1_task3_simulation_results.png', dpi=150, bbox_inches='tight')
    print("  → Sauvegardé : part1_task3_simulation_results.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 4 : Instructions pour le profiling avec kernprof / line_profiler
# ─────────────────────────────────────────────────────────────────────────────
def task4_profiling_instructions():
    """
    Explique comment profiler la fonction jacobi avec kernprof.

    kernprof est un outil qui mesure le temps ligne par ligne dans une
    fonction Python. On ajoute @profile devant la fonction à analyser,
    puis on exécute kernprof.
    """
    print("\n" + "=" * 60)
    print("TÂCHE 4 – Profiling avec kernprof")
    print("=" * 60)
    print("""
Pour profiler la fonction jacobi() ligne par ligne :

  1. Crée un fichier 'jacobi_profile.py' contenant :

        from simulate import load_data, summary_stats
        import numpy as np

        LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

        @profile                         ← décorateur magique de kernprof
        def jacobi(u, interior_mask, max_iter, atol=1e-6):
            u = np.copy(u)
            for i in range(max_iter):
                u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:]
                              + u[:-2, 1:-1] + u[2:, 1:-1])
                u_new_interior = u_new[interior_mask]
                delta = np.abs(u[1:-1, 1:-1][interior_mask]
                               - u_new_interior).max()
                u[1:-1, 1:-1][interior_mask] = u_new_interior
                if delta < atol:
                    break
            return u

        if __name__ == '__main__':
            with open(LOAD_DIR + 'building_ids.txt') as f:
                bid = f.readline().strip()
            u0, mask = load_data(LOAD_DIR, bid)
            jacobi(u0, mask, 100)          ← seulement 100 itérations pour aller vite

  2. Lance :
        kernprof -l -v jacobi_profile.py

  Résultat attendu (exemple simplifié) :
  ┌─────────────────────────────────────────────────────────────┐
  │ Line  Hits    Time   Per Hit  % Time  Contents              │
  │  18  20000  120000  6.0      2.0%   for i in range(...)    │
  │  20  20000 3000000  150.0   50.0%   u_new = 0.25 * (...)   │  ← DOMINANT
  │  21  20000  800000   40.0   13.0%   u_new_interior = ...   │
  │  22  20000 1500000   75.0   25.0%   delta = np.abs(...)    │  ← LENT
  │  23  20000  400000   20.0    7.0%   u[1:-1,1:-1][mask]=..  │
  └─────────────────────────────────────────────────────────────┘

  Interprétation :
  • Ligne 20 (calcul u_new) : la plus coûteuse → opération NumPy vectorisée
    sur une grille 514×514 = 264 000 points, à chaque itération.
  • Ligne 22 (calcul delta) : coûteux car il faut trouver le MAX sur ~200k points.
  • Ligne 21 (indexation masque) : extraction des valeurs intérieures.
  • Total : ~20 000 itérations × ~3 opérations lourdes = TRÈS lent en pur Python.

  → CONCLUSION : le goulot d'étranglement est la boucle Jacobi elle-même.
    Stratégies d'optimisation :
      (a) Paralléliser sur plusieurs CPU (Parties 2)
      (b) Compiler avec Numba JIT (Partie 3)
      (c) Passer sur GPU avec CUDA/CuPy (Partie 4)
""")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Chargement de la liste des IDs de bâtiments
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print(f"Dataset : {len(building_ids)} bâtiments disponibles.")
    print(f"Exemples d'IDs : {building_ids[:5]}")

    n_viz = 4   # nombre de bâtiments à visualiser

    task1_visualize_input(building_ids, n=n_viz)
    sec_per_building = task2_timing(building_ids, n_test=10)
    task3_visualize_results(building_ids, n=n_viz)
    task4_profiling_instructions()

    print("\n✓ Partie 1 terminée. Regarde les fichiers PNG générés.")
