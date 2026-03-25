"""
=============================================================================
PARTIE 3  –  Optimisation CPU avec Numba JIT
=============================================================================
Tâche couverte : 7 (réécriture de jacobi avec Numba JIT)

Expert imaginaire : Carla – Ingénieure Performance / Numba / Architecture CPU
=============================================================================

CONCEPT CLÉ POUR DÉBUTANT :
  Python est lent car il est interprété : chaque ligne est traduite en
  instructions machine UNE PAR UNE à chaque exécution.

  Numba (@njit) compile la fonction en code machine NATIF la première fois
  qu'elle est appelée (JIT = Just-In-Time compilation). Les appels suivants
  utilisent ce code ultra-rapide directement.

  POURQUOI LES BOUCLES EXPLICITES SONT MEILLEURES AVEC NUMBA :
  Avec NumPy, on fait des opérations sur des TABLEAUX ENTIERS. C'est efficace
  car NumPy utilise du code C en dessous. Mais avec Numba, les boucles Python
  EXPLICITES (for i in ...) deviennent aussi rapides que du C/Fortran, et
  permettent un meilleur contrôle du cache CPU.

  CACHE CPU :
  Le processeur a un petit espace mémoire ultra-rapide (cache L1/L2/L3).
  Si on accède aux données dans l'ORDRE (ligne par ligne), le cache est
  efficace (cache-friendly). Si on saute partout en mémoire, le cache
  se vide et le CPU attend → lent.

  En C, les tableaux 2D sont stockés en mémoire LIGNE PAR LIGNE (row-major).
  Donc parcourir [i][j] avec j qui varie le plus vite = accès séquentiel = RAPIDE.

Usage :
    python part3_numba.py <N>
    Ex :  python part3_numba.py 10
"""

from os.path import join
import sys
import time

import numpy as np
from numba import njit, prange

from simulate import load_data, jacobi as jacobi_reference, summary_stats

# ─────────────────────────────────────────────────────────────────────────────
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4
N_TOTAL  = 4571


# ─────────────────────────────────────────────────────────────────────────────
# VERSION NUMBA JIT  –  boucles explicites
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-4):
    """
    Version compilée JIT de l'algorithme de Jacobi.

    DIFFÉRENCES AVEC LA VERSION NUMPY :
    ─────────────────────────────────────────────────────────────────────────
    • On utilise des boucles FOR explicites sur i,j au lieu d'opérations
      sur tableaux entiers. Avec Numba, ces boucles compilées sont aussi
      rapides (voire plus) que NumPy.

    • On n'a PAS besoin d'allouer un tableau u_new complet à chaque itération.
      On utilise un tableau temporaire u_old (copie) puis on met à jour u en place.
      → Moins d'allocations mémoire → plus rapide.

    ACCÈS MÉMOIRE CACHE-FRIENDLY :
    ─────────────────────────────────────────────────────────────────────────
    Le tableau u est stocké ligne par ligne en mémoire :
        [u[0,0], u[0,1], u[0,2], ..., u[0,513], u[1,0], u[1,1], ...]
    La boucle intérieure sur 'j' (colonne) varie d'abord → accès SÉQUENTIEL
    → Le CPU peut pré-charger les données dans le cache avant qu'on en ait besoin
    → Cache-hits élevés → exécution rapide.

    Si on avait la boucle sur 'i' à l'intérieur (i varie vite), on accèderait
    à u[0,j], u[1,j], u[2,j]... qui sont espacés de 514 cases en mémoire → lent.

    PARAMÈTRES :
        u             : grille 514×514 avec conditions aux bords déjà fixées
        interior_mask : masque booléen 512×512 (True = point intérieur)
        max_iter      : nombre maximum d'itérations
        atol          : tolérance de convergence (critère d'arrêt)
    """
    # Dimensions de la grille intérieure (sans les bords)
    rows = interior_mask.shape[0]   # = 512
    cols = interior_mask.shape[1]   # = 512

    # Copie de travail pour ne pas modifier u en place pendant le calcul
    # (propriété nécessaire de la méthode de Jacobi vs Gauss-Seidel)
    u_old = np.empty_like(u)

    for iteration in range(max_iter):

        # ── Copier u → u_old ─────────────────────────────────────────────
        # On doit conserver les anciennes valeurs pendant qu'on met à jour
        for i in range(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        # ── Itération Jacobi + calcul de la convergence ───────────────────
        max_delta = 0.0  # on cherche le maximum de |u_new - u_old|

        # BOUCLE CACHE-FRIENDLY : i (ligne) à l'extérieur, j (colonne) à l'intérieur
        for i in range(rows):       # i de 0 à 511 (indices grille intérieure)
            for j in range(cols):   # j de 0 à 511

                if interior_mask[i, j]:
                    # Indices dans u : les bords ajoutent +1 décalage
                    # u[i+1, j+1] = point central
                    # Voisins : gauche u[i+1, j], droite u[i+1, j+2],
                    #           haut   u[i, j+1],  bas    u[i+2, j+1]
                    val = 0.25 * (
                        u_old[i+1, j  ]   # voisin gauche
                      + u_old[i+1, j+2]   # voisin droite
                      + u_old[i,   j+1]   # voisin haut
                      + u_old[i+2, j+1]   # voisin bas
                    )

                    # Calcul de la variation (pour le critère de convergence)
                    diff = val - u_old[i+1, j+1]
                    if diff < 0.0:
                        diff = -diff   # valeur absolue manuelle (pas math.fabs dans @njit)
                    if diff > max_delta:
                        max_delta = diff

                    # Mise à jour du point
                    u[i+1, j+1] = val

        # ── Vérification de la convergence ────────────────────────────────
        if max_delta < atol:
            break  # la grille ne change plus → solution convergée

    return u


# ─────────────────────────────────────────────────────────────────────────────
# VERSION NUMBA JIT PARALLÈLE  –  bonus avec prange
# ─────────────────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True)
def jacobi_numba_parallel(u, interior_mask, max_iter, atol=1e-4):
    """
    Version Numba avec parallélisme interne (prange).

    prange = parallel range : les itérations de la boucle externe sont
    distribuées automatiquement sur tous les cœurs CPU disponibles.

    ATTENTION : avec le parallélisme, on NE PEUT PAS avoir de critère
    d'arrêt anticipé facilement (race condition sur max_delta), donc on
    utilise une réduction manuelle et on tolère une légère perte de précision
    dans le critère d'arrêt.
    """
    rows = interior_mask.shape[0]
    cols = interior_mask.shape[1]
    u_old = np.empty_like(u)

    for iteration in range(max_iter):
        # Copie parallèle
        for i in prange(rows + 2):
            for j in range(cols + 2):
                u_old[i, j] = u[i, j]

        max_delta = 0.0

        # Boucle parallèle sur les lignes
        for i in prange(rows):
            local_delta = 0.0
            for j in range(cols):
                if interior_mask[i, j]:
                    val = 0.25 * (
                        u_old[i+1, j  ]
                      + u_old[i+1, j+2]
                      + u_old[i,   j+1]
                      + u_old[i+2, j+1]
                    )
                    diff = val - u_old[i+1, j+1]
                    if diff < 0.0:
                        diff = -diff
                    if diff > local_delta:
                        local_delta = diff
                    u[i+1, j+1] = val

            # Réduction manuelle du max (thread-safe avec Numba)
            if local_delta > max_delta:
                max_delta = local_delta

        if max_delta < atol:
            break

    return u


# ─────────────────────────────────────────────────────────────────────────────
# WARM-UP : déclencher la compilation JIT avant le benchmark
# ─────────────────────────────────────────────────────────────────────────────
def warmup_numba(u_sample, mask_sample):
    """
    La première exécution de @njit compile le code → prend ~10s.
    On fait un warm-up sur 1 itération pour que le benchmark soit juste.
    """
    print("  Compilation JIT en cours (warm-up)...", end=' ', flush=True)
    t0 = time.perf_counter()
    _ = jacobi_numba(u_sample.copy(), mask_sample, max_iter=1)
    _ = jacobi_numba_parallel(u_sample.copy(), mask_sample, max_iter=1)
    print(f"terminé en {time.perf_counter()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK : comparaison référence vs Numba
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(building_ids, n_test):
    """
    Compare les temps d'exécution :
      (a) Référence NumPy (script fourni par le cours)
      (b) Numba JIT séquentiel
      (c) Numba JIT parallèle (bonus)
    """
    print(f"\n{'Méthode':<25} | {'Temps (s)':>10} | {'Speed-up':>10}")
    print("-" * 52)

    ids = building_ids[:n_test]
    results = {}

    # ── (a) Référence NumPy ───────────────────────────────────────────────
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_reference(u0, mask, MAX_ITER, ABS_TOL)
    t_ref = time.perf_counter() - t0
    print(f"{'(a) Référence NumPy':<25} | {t_ref:>10.2f} | {'1.00x':>10}")
    results['reference'] = t_ref

    # ── (b) Numba JIT séquentiel ──────────────────────────────────────────
    # Warm-up sur le premier bâtiment
    u_sample, mask_sample = load_data(LOAD_DIR, ids[0])
    warmup_numba(u_sample, mask_sample)

    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
    t_numba = time.perf_counter() - t0
    speedup = t_ref / t_numba
    print(f"{'(b) Numba JIT séquentiel':<25} | {t_numba:>10.2f} | {speedup:>9.2f}x")
    results['numba_seq'] = t_numba

    # ── (c) Numba JIT parallèle ───────────────────────────────────────────
    t0 = time.perf_counter()
    for bid in ids:
        u0, mask = load_data(LOAD_DIR, bid)
        _ = jacobi_numba_parallel(u0.copy(), mask, MAX_ITER, ABS_TOL)
    t_numba_par = time.perf_counter() - t0
    speedup_par = t_ref / t_numba_par
    print(f"{'(c) Numba JIT parallèle':<25} | {t_numba_par:>10.2f} | {speedup_par:>9.2f}x")
    results['numba_par'] = t_numba_par

    # ── Extrapolation ──────────────────────────────────────────────────────
    print("\n  Extrapolation à 4571 bâtiments :")
    for label, t in [('Référence', t_ref), ('Numba seq', t_numba),
                     ('Numba par', t_numba_par)]:
        t_est = (t / n_test) * N_TOTAL
        print(f"    {label:<12} : {t_est/60:.1f} min ({t_est/3600:.2f}h)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION DE CORRECTION (résultats identiques ?)
# ─────────────────────────────────────────────────────────────────────────────
def verify_correctness(building_ids, n_check=3):
    """
    Vérifie que la version Numba donne les mêmes résultats que la référence.
    On calcule la différence maximale entre les deux grilles résultat.
    """
    print("\n─── Vérification de correction ─────────────────────────────")
    for bid in building_ids[:n_check]:
        u0, mask = load_data(LOAD_DIR, bid)

        u_ref   = jacobi_reference(u0,       mask, MAX_ITER, ABS_TOL)
        u_numba = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)

        max_diff = np.abs(u_ref - u_numba).max()
        status   = "✓ OK" if max_diff < 1e-3 else "✗ ERREUR"
        print(f"  Bâtiment {bid} : diff max = {max_diff:.2e}  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    print("=" * 55)
    print("  PARTIE 3 – Optimisation Numba JIT")
    print("=" * 55)
    print(f"  Bâtiments de test : {N}")

    verify_correctness(building_ids)
    results = benchmark(building_ids, n_test=N)

    # Affichage CSV des résultats avec Numba
    print("\n─── Résultats CSV (Numba JIT) ──────────────────────────────")
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid in building_ids[:N]:
        u0, mask = load_data(LOAD_DIR, bid)
        u = jacobi_numba(u0.copy(), mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
