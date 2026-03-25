"""
jacobi_profile.py  –  Script à utiliser avec kernprof pour le profiling ligne-à-ligne.

Utilisation :
    kernprof -l -v jacobi_profile.py

kernprof instrumente chaque ligne de la fonction marquée @profile
et mesure combien de temps est passé sur chaque ligne.
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


@profile  # ← décorateur SPÉCIAL kernprof (ne pas importer, kernprof l'injecte)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """
    Fonction Jacobi profilée ligne par ligne.
    On utilise seulement 200 itérations pour ne pas attendre trop longtemps.
    """
    u = np.copy(u)                         # ligne 1 : copie de la grille
    for i in range(max_iter):              # ligne 2 : boucle principale
        # Calcul de la moyenne des 4 voisins pour tous les points intérieurs
        # (opération matricielle NumPy = rapide, mais appelée 200× !)
        u_new = 0.25 * (                   # ligne 3 : ← LIGNE LA PLUS LENTE
            u[1:-1, :-2]                   #   voisin gauche
          + u[1:-1, 2:]                    #   voisin droite
          + u[:-2, 1:-1]                   #   voisin haut
          + u[2:, 1:-1]                    #   voisin bas
        )
        # Extraction des valeurs intérieures (indexation booléenne)
        u_new_interior = u_new[interior_mask]  # ligne 4 : extraction masque

        # Calcul de la variation maximale (critère de convergence)
        delta = np.abs(                        # ligne 5 : ← 2ÈME PLUS LENTE
            u[1:-1, 1:-1][interior_mask] - u_new_interior
        ).max()

        # Mise à jour uniquement des points intérieurs
        u[1:-1, 1:-1][interior_mask] = u_new_interior  # ligne 6 : mise à jour

        if delta < atol:                       # ligne 7 : test de convergence
            break
    return u


if __name__ == '__main__':
    # Chargement d'un seul bâtiment pour le profiling
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        bid = f.readline().strip()

    u0, interior_mask = load_data(LOAD_DIR, bid)

    # On profile seulement 200 itérations pour aller vite
    print(f"Profiling de la fonction jacobi sur bâtiment {bid}...")
    print("(200 itérations)")
    jacobi(u0, interior_mask, max_iter=200)
    print("Profiling terminé.")
