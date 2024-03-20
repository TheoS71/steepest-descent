"""Minimisez une molécule d'eau selon l'algorithme de steepest descent.
   Ici deux méthodes : Gradient analytique et gradient numérique

Usage:
======
    python algo_minimisation_steepest-descent.py argument1 argument2

    argument1: Le nom de votre fichier contenant la molécule d'eau au format pdb
    argument2: La méthode du gradient utilisée : analytical | numerical
"""

__authors__ = ("Théo Serralta")
__contact__ = ("theo.serralta@gmail.com")
__copyright__ = "MIT"
__date__ = "2023-03-20"
__version__= "1.0"

#Definitions librairies
import numpy as np
import math as m
import sys

#Fonction extraction des coordonées
def extract_coord_pdb(pdbfile):
    """Extraction des coordonées.

    Cette fonction sert à extraire les coordonées d'une molécule dans un fichier pdb.

    Parameters
    ----------
    pdbfile : str
        Le fichier contenant la ou les molecules.

    Returns
    -------
    tab
        Un tableau numpy contenant les coordonnées des atomes.
    """
    coordinates = []
    with open(pdbfile, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                x = float(line.split()[6])
                y = float(line.split()[7])
                z = float(line.split()[8])
                coordinates.extend([x, y, z])
    return np.array(coordinates)

#Fonction calcul longueurs liaisons et angle
def calculate_bond_lengths_and_angle(coords):
    """Calcul longueur de liaison et angle.

    Cette fonction sert à calculer les longueurs de liaison ainsi que l'angle de la molécule.

    Parameters
    ----------
    coords : numpy.ndarray
        Les coordonnées des trois atomes sous forme d'un tableau numpy.

    Returns
    -------
    tuple
        Un tuple contenant les longueurs des liaisons et l'angle entre les atomes.
    """
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = coords

    l1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    l2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
    angle = m.acos(((x1 - x2)*(x3-x2) + (y1-y2)*(y3-y2) + (z1-z2)*(z3-z2))/(l1*l2))

    return l1, l2, angle

#Fonction calcul energie liaisons
def calculate_bond_energy(l1, l2):
    """Calcul de l'énergie de liaison.

    Cette fonction calcule l'énergie de liaison entre deux atomes.

    Parameters
    ----------
    l1 : float
        Longueur de la première liaison.
    l2 : float
        Longueur de la deuxième liaison.

    Returns
    -------
    float
        L'énergie de liaison.
    """
    E_bond = ((k_bond/2)*(l1-i_eq)**2) + ((k_bond/2)*(l2-i_eq)**2)
    return E_bond

#Fonction energie angle
def calculate_angle_energy(angle):
    """Calcul de l'énergie d'angle.

    Cette fonction calcule l'énergie associée à un angle entre trois atomes.

    Parameters
    ----------
    angle : float
        L'angle entre les atomes en radians.

    Returns
    -------
    float
        L'énergie de l'angle.
    """
    E_angle =  ((k_angle/2)*(angle-(teta_eq*m.pi)/180)**2)
    return E_angle

#Fonction calcul energie totale
def calculate_E(E_bond, E_angle):
    """Calcul de l'énergie totale.

    Cette fonction calcule l'énergie totale de la molécule en additionnant
    l'énergie de liaison et l'énergie d'angle.

    Parameters
    ----------
    E_bond : float
        L'énergie de liaison.
    E_angle : float
        L'énergie d'angle.

    Returns
    -------
    float
        L'énergie totale de la molécule.
    """
    E = E_bond + E_angle
    return E

#Fonction calcul gradient numérique
def calculate_gradient_numerical(coordinates, h):
    """Calcul du gradient numérique.

    Cette fonction calcule le gradient numérique de l'énergie par rapport
    aux coordonnées atomiques en utilisant une méthode de différences finies.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Les coordonnées des atomes de la molécule.
    h : float
        La taille du pas pour le calcul numérique du gradient.

    Returns
    -------
    numpy.ndarray
        Le gradient de l'énergie par rapport aux coordonnées atomiques.
    """
    gradient = []
    
    for i in range(len(coordinates)):
        coords_plus_h = coordinates.copy()
        coords_plus_h[i] += h
        coords_minus_h = coordinates.copy()
        coords_minus_h[i] -= h
        
        l1_plus_h, l2_plus_h, angle_plus_h = calculate_bond_lengths_and_angle(coords_plus_h)
        E_bond_plus_h = calculate_bond_energy(l1_plus_h, l2_plus_h)
        E_angle_plus_h = calculate_angle_energy(angle_plus_h)
        E_plus_h = calculate_E(E_bond_plus_h, E_angle_plus_h)
        
        l1_minus_h, l2_minus_h, angle_minus_h = calculate_bond_lengths_and_angle(coords_minus_h)
        E_bond_minus_h = calculate_bond_energy(l1_minus_h, l2_minus_h)
        E_angle_minus_h = calculate_angle_energy(angle_minus_h)
        E_minus_h = calculate_E(E_bond_minus_h, E_angle_minus_h)
        
        dE_dcoord = (E_plus_h - E_minus_h) / (2 * h)
        gradient.append(dE_dcoord)
    
    return np.array(gradient)

#Fonction calcul gradient analytique
def calculate_gradient_analytical(coordinates):
    """Calcul du gradient analytique.

    Cette fonction calcule le gradient analytique de l'énergie totale par rapport
    aux coordonnées atomiques de la molécule.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Les coordonnées des atomes de la molécule.

    Returns
    -------
    numpy.ndarray
        Le gradient de l'énergie par rapport aux coordonnées atomiques.
    """
    gradient = []
    
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = coordinates
    
    # Recuperation des longueurs de liaison et angle
    l1, l2, angle = calculate_bond_lengths_and_angle(coordinates)

    
    # Dérivées partielles de l'énergie de liaison par rapport aux coordonnées atomiques
    dE_bond_dx1 = k_bond * (l1 - i_eq) * (x1 - x2) / l1
    dE_bond_dy1 = k_bond * (l1 - i_eq) * (y1 - y2) / l1
    dE_bond_dz1 = k_bond * (l1 - i_eq) * (z1 - z2) / l1
    dE_bond_dx3 = k_bond * (l2 - i_eq) * (x3 - x2) / l2
    dE_bond_dy3 = k_bond * (l2 - i_eq) * (y3 - y2) / l2
    dE_bond_dz3 = k_bond * (l2 - i_eq) * (z3 - z2) / l2
    
    # Calcul du facteur commun pour les dérivées d'angle
    common_factor_angle = k_angle * (angle - (teta_eq * m.pi) / 180) / (l1 * l2 * m.sin(angle))

    # Dérivées partielles de l'énergie d'angle par rapport aux coordonnées atomiques
    dE_angle_dx1 = common_factor_angle * (x1 - x2) * (y3 - y2)
    dE_angle_dy1 = common_factor_angle * (y1 - y2) * (x3 - x2)
    dE_angle_dz1 = common_factor_angle * (z1 - z2) * (x3 - x2)
    dE_angle_dx2 = common_factor_angle * (x3 - x2) * (y1 - y2)
    dE_angle_dy2 = common_factor_angle * (y3 - y2) * (x1 - x2)
    dE_angle_dz2 = common_factor_angle * (z3 - z2) * (x1 - x2)

    # Dérivées partielles du gradient
    dE_dx1 = dE_bond_dx1 + dE_angle_dx1
    dE_dy1 = dE_bond_dy1 + dE_angle_dy1
    dE_dz1 = dE_bond_dz1 + dE_angle_dz1
    dE_dx2 = -dE_bond_dx1 - dE_bond_dx3 + dE_angle_dx1 - dE_angle_dx2
    dE_dy2 = -dE_bond_dy1 - dE_bond_dy3 + dE_angle_dy1 - dE_angle_dy2
    dE_dz2 = -dE_bond_dz1 - dE_bond_dz3 + dE_angle_dz1 - dE_angle_dz2
    dE_dx3 = dE_bond_dx3 + dE_angle_dx2
    dE_dy3 = dE_bond_dy3 + dE_angle_dy2
    dE_dz3 = dE_bond_dz3 + dE_angle_dz2
    
    # Ajout des dérivées partielles au gradient
    gradient.append(dE_dx1)
    gradient.append(dE_dy1)
    gradient.append(dE_dz1)
    gradient.append(dE_dx2)
    gradient.append(dE_dy2)
    gradient.append(dE_dz2)
    gradient.append(dE_dx3)
    gradient.append(dE_dy3)
    gradient.append(dE_dz3)
    
    return np.array(gradient)

#Fonction choix du gradient
def calculate_gradient(coordinates, h, use_analytical=True):
    """
    Calcul du gradient de l'énergie par rapport aux coordonnées atomiques.

    Cette fonction calcule le gradient de l'énergie par rapport aux coordonnées atomiques
    de la molécule en utilisant soit une méthode analytique, soit une méthode numérique.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Les coordonnées des atomes de la molécule.
    h : float
        La valeur du pas utilisée pour le calcul numérique du gradient.
    use_analytical : bool, optional
        Indique si le gradient analytique doit être utilisé. Par défaut, True.

    Returns
    -------
    numpy.ndarray
        Le gradient de l'énergie par rapport aux coordonnées atomiques.
    """
    if use_analytical:
        return calculate_gradient_analytical(coordinates)
    else:
        return calculate_gradient_numerical(coordinates, h)

#Fonction calculs nouvelles coordonnées
def update_coordinates(coordinates, gradient, step_size):
    """Mise à jour des coordonnées.

    Cette fonction met à jour les coordonnées atomiques en fonction du gradient
    et de la taille du pas donnés.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Les coordonnées des atomes avant la mise à jour.
    gradient : numpy.ndarray
        Le gradient de l'énergie par rapport aux coordonnées atomiques.
    step_size : float
        La taille du pas pour la mise à jour des coordonnées.

    Returns
    -------
    numpy.ndarray
        Les coordonnées des atomes après la mise à jour.
    """
    updated_coordinates = coordinates - step_size * gradient
    return updated_coordinates

#Fonction minimisation gradient numérique
def minimize_energy_numerical(pdbfile, h, step_size, threshold, max_iterations, k_bond, i_eq, k_angle, teta_eq, use_analytical=False):
    """Minimisation de l'énergie d'une molécule en utilisant le gradient numérique.

    Cette fonction minimise l'énergie d'une molécule en utilisant le gradient numérique.
    Elle calcule le gradient de l'énergie par rapport aux coordonnées atomiques en utilisant
    une méthode de différences finies.

    Parameters
    ----------
    pdbfile : str
        Le chemin vers le fichier pdb contenant les coordonnées de la molécule.
    h : float
        La valeur du pas utilisée pour le calcul numérique du gradient.
    step_size : float
        La taille du pas pour la mise à jour des coordonnées.
    threshold : float
        Le seuil de convergence pour le GRMS. L'optimisation s'arrête lorsque le GRMS est inférieur à ce seuil.
    max_iterations : int
        Le nombre maximal d'itérations autorisé avant d'arrêter l'optimisation.
    k_bond : float
        La constante de force de l'énergie de liaison.
    i_eq : float
        La distance à l'équilibre pour les liaisons.
    k_angle : float
        La constante de force de l'énergie d'angle.
    teta_eq : float
        L'angle à l'équilibre en degrés.
    use_analytical : bool, optional
        Indique si le gradient analytique doit être utilisé. Par défaut, False.

    Returns
    -------
    None
        La fonction affiche les résultats finaux de l'optimisation.
    """
    initial_coordinates = extract_coord_pdb(pdbfile)

    for iteration in range(max_iterations):
        gradient = calculate_gradient_numerical(initial_coordinates, h)
        new_coordinates = update_coordinates(initial_coordinates, gradient, step_size)
        
        # Calcul du GRMS
        grms = np.linalg.norm(gradient) / np.sqrt(len(initial_coordinates))
        
        # Vérification du critère d'arrêt
        if grms < threshold:
            print("Convergence atteinte. Arrêt de l'optimisation.")
            break
        
        # Mise à jour des coordonnées pour la prochaine itération
        initial_coordinates = new_coordinates
    
    else:
        print("Nombre maximal d'itérations atteint sans convergence.")
    
    # Affichage des résultats finaux
    final_lengths = calculate_bond_lengths_and_angle(new_coordinates)
    new_energy_bond = calculate_bond_energy(final_lengths[0], final_lengths[1])
    new_angle_energy = calculate_angle_energy(final_lengths[2])
    print(f"Iteration {iteration+1} - GRMS: {grms}")
    print(f"Énergie finale : {new_energy_bond + new_angle_energy} kcal/mol")
    print(f"Angle après minimisation : {final_lengths[2] * 180 / m.pi:.4f} degrés")
    print(f"Longueur de liaison l1 après minimisation : {final_lengths[0]:.4f} Å")
    print(f"Longueur de liaison l2 après minimisation : {final_lengths[1]:.4f} Å")

#Fonction minimisation gradient analytique
def minimize_energy_analytical(pdbfile, threshold, max_iterations, k_bond, i_eq, k_angle, teta_eq, use_analytical=True):
    """    Minimisation de l'énergie d'une molécule en utilisant le gradient analytique.

    Cette fonction minimise l'énergie d'une molécule en utilisant le gradient analytique.
    Elle utilise des dérivées analytiques pour calculer le gradient de l'énergie par rapport
    aux coordonnées atomiques.

    Parameters
    ----------
    pdbfile : str
        Le chemin vers le fichier pdb contenant les coordonnées de la molécule.
    h : float
        La valeur du pas utilisée pour le calcul numérique du gradient.
    threshold : float
        Le seuil de convergence pour le GRMS. L'optimisation s'arrête lorsque le GRMS est inférieur à ce seuil.
    max_iterations : int
        Le nombre maximal d'itérations autorisé avant d'arrêter l'optimisation.
    k_bond : float
        La constante de force de l'énergie de liaison.
    i_eq : float
        La distance à l'équilibre pour les liaisons.
    k_angle : float
        La constante de force de l'énergie d'angle.
    teta_eq : float
        L'angle à l'équilibre en degrés.

    Returns
    -------
    None
        La fonction affiche les résultats finaux de l'optimisation.
    """
    initial_coordinates = extract_coord_pdb(pdbfile)
    step_size = 10**-3  # Définissez ici votre taille de pas
    
    for iteration in range(max_iterations):
        gradient = calculate_gradient_analytical(initial_coordinates)
        new_coordinates = update_coordinates(initial_coordinates, gradient, step_size)
        
        # Calcul du GRMS
        grms = np.linalg.norm(gradient) / np.sqrt(len(initial_coordinates))
        
        #print(f"Iteration {iteration+1} - GRMS: {grms}")
        
        # Vérification du critère d'arrêt
        if grms < threshold:
            print("Convergence atteinte. Arrêt de l'optimisation.")
            break
        
        # Mise à jour des coordonnées pour la prochaine itération
        initial_coordinates = new_coordinates
    
    else:
        print("Nombre maximal d'itérations atteint sans convergence.")
    
    # Affichage des résultats finaux
    final_lengths = calculate_bond_lengths_and_angle(new_coordinates)
    new_energy_bond = calculate_bond_energy(final_lengths[0], final_lengths[1])
    new_angle_energy = calculate_angle_energy(final_lengths[2])
    print(f"Iteration {iteration+1} - GRMS: {grms}")
    print(f"Énergie finale : {new_energy_bond + new_angle_energy} kcal/mol")
    print(f"Angle après minimisation : {final_lengths[2] * 180 / m.pi:.4f} degrés")
    print(f"Longueur de liaison l1 après minimisation : {final_lengths[0]:.4f} Å")
    print(f"Longueur de liaison l2 après minimisation : {final_lengths[1]:.4f} Å")


# Programme principal
if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 script.py <pdbfile> [analytical|numerical]")
        sys.exit(1)
    
    pdbfile = sys.argv[1]
    method = "analytical"  # Par défaut, utilise le gradient analytique
    if len(sys.argv) == 3:
        method = sys.argv[2]
        if method not in ["analytical", "numerical"]:
            print("Invalid method argument. Please use 'analytical' or 'numerical'.")
            sys.exit(1)
    
    h = 10**-6 #Condition de convergence
    step_size = 10**-3 # Pas du gradient
    threshold = 10**-10  # Seuil pour GRMS
    max_iterations = 1000  # Nombre maximal d'itérations pour éviter les boucles infinies
    k_bond = 450 #Constante de force énergie de liaison
    i_eq = 0.9572 #Distance à l'équilibre
    k_angle = 55 #Constante de force énergie d'angle
    teta_eq = 104.5200 #Angle à l'équilibre

    use_analytical = True if method == "analytical" else False

    if use_analytical:
        minimize_energy_analytical(pdbfile, threshold, max_iterations, k_bond, i_eq, k_angle, teta_eq)
    else:
        minimize_energy_numerical(pdbfile, h, step_size, threshold, max_iterations, k_bond, i_eq, k_angle, teta_eq)

