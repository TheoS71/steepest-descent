"""Steepest Gradient Descent minimization of water molecules.

A script that minimizes the energy of a system of water 
molecules using Steepest Gradient descent.

Usage:
======
    python mini2.py filename

    filename: the name of the file containing atomic coordinates.
"""

__authors__ = "Nadezhda Zhukova"
__contact__ = "nadezhda.zhukova@inserm.fr"
__copyright__ = "MIT"
__date__ = "2024-03-20"
__version__ = "1.0.0"


import colorsys
import copy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

############################## DEFINE CONSTANTS ##############################
k_bond = 450  # kcal//mol^(-1)/A^(-2) 
l_0 = 0.9572  # A
k_angle = 55  # kcal/mol^(-1)/rad^(-2) 
theta_0 = 104.52 * np.pi / 180 # rad
epsilons = {"HH": 0.0460, "OH": 0.0836, "OO": 0.1521} # kcal/mol^(-1)
sigmas = {"HH": 0.4490, "OH": 1.9927, "OO": 3.5364}   # A
charges = {"H": 0.4170,"O": -0.8340} # e
##############################################################################


def read_coordinates(filename):
    """
    Reads atomic coordinates from a file and returns a dictionary.

    Args:
    filename (str): The name of the file containing atomic coordinates.

    Returns:
    dict: A dictionary where keys are atom names and values are coordinates.
    """
    coordinates = {}
    with open(filename, "r") as file:
        for line in file:
            fields = line.split()
            atom_name = f"{fields[0]}_{fields[1]}"
            coords = np.array([float(coord) for coord in fields[3:6]])
            coordinates[atom_name] = coords
    return coordinates


def generate_colors(n):
    """
    Generate a list of n colors.

    Args:
        n (int): The number of colors to generate.

    Returns:
        list : A list of n RGB colors.
    """
    colors = []
    # HSV color space
    hues = [i / n for i in range(n)]
    # Convert HSV to an RGB color
    for hue in hues:
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)
    return colors


def plot_molecule(coordinates):
    """
    Plot a molecule using atomic coordinates.

    Args:
        coordinates (dict): A dictionary containing atomic coordinates.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = generate_colors(len(coordinates)//3)
    for i, (atom_label, coord) in enumerate(coordinates.items()):
        atom_name = atom_label.split("_")[1]
        ax.scatter(coord[0], coord[1], coord[2], label=atom_label, color=colors[i//3], marker="o", s=1000, alpha=0.5)
        ax.text(coord[0], coord[1], coord[2], atom_name, fontsize=10)
        if i % 3 == 0:
            o_coord = coord
        else:
            h_coord = coord
            ax.plot([o_coord[0], h_coord[0]], [o_coord[1], h_coord[1]], [o_coord[2], h_coord[2]], "k--")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def calculate_dist_and_angle(coordinates):
    """
    Calculates OH bond lengths and angles for water molecules.

    Args:
    coordinates (dict): A dictionary containing atomic coordinates.

    Returns:
    tuple: Two lists, containing OH bond lengths and angles.
    """
    OH_bonds = []
    thetas = []
    atom_names = list(coordinates.keys())
    # Assuming the coordinates are ordered OW, HW1, HW2 for each water molecule
    for i in range(0, len(atom_names), 3):
        OW, HW1, HW2 = [coordinates[atom_names[i + j]] for j in range(3)]
        OH_bonds.extend([np.linalg.norm(OW - HW) for HW in [HW1, HW2]])
        v1 = HW1 - OW
        v2 = HW2 - OW
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(cos_theta)
        thetas.append(theta)
    return OH_bonds, thetas


def calculate_bond_energy(l_OH, k_bond, l_0):
    """
    Calculate the bond energy using the harmonic potential.

    Args:
        l_OH (list): A list of OH bond lengths.
        k_bond (float): The bond spring constant.
        l_0 (float): The equilibrium bond length.

    Returns:
       float : The bond energy.
    """
    bond_energy = 0.5 * k_bond * sum((l - l_0) ** 2 for l in l_OH)
    return bond_energy


def calculate_angle_energy(thetas, k_angle, theta_0):
    angle_energy = 0.5 * k_angle * sum((theta - theta_0) ** 2 for theta in thetas)
    return angle_energy


def calculate_vdw_energy(coordinates, epsilons, sigmas):
    """
    Calculate the Van der Waals energy using the Lennard-Jones potential.

    Args:
        coordinates (dict): A dictionary with atom labels and their coordinates.
        epsilons (dict): A dictionary containing epsilon values for atom pairs.
        sigmas (dict): A dictionary containing sigma values for atom pairs.

    Returns:
        float: The van der Waals energy.
    """
    vdw_energy = 0

    for atom_label_i, coords_i in coordinates.items():
        for atom_label_j, coords_j in coordinates.items():
            # Atom types (O or H)
            atom_type_i = atom_label_i.split("_")[1][0]
            atom_type_j = atom_label_j.split("_")[1][0]
            # Molecule number
            mol_nb_i = atom_label_i.split("_")[0][0]
            mol_nb_j = atom_label_j.split("_")[0][0]
            # Calculate interactions only between atoms of different molecules
            if mol_nb_i != mol_nb_j:
                r_ij = np.linalg.norm(coords_i - coords_j)
                atom_types = f"{atom_type_i}{atom_type_j}"
                try:
                    epsilon = epsilons[atom_types]
                    sigma = sigmas[atom_types]
                except KeyError:
                    # Try swapping atom types
                    atom_types = f"{atom_type_j}{atom_type_i}"  
                    epsilon = epsilons[atom_types]
                    sigma = sigmas[atom_types]
                vdw_energy += 4 * epsilon * ((sigma / r_ij) ** 12 - (sigma / r_ij) ** 6)
    return vdw_energy



def calculate_coulomb_energy(coordinates, charges, f=332.0716):
    """
    Calculate the Coulomb energy using atomic coordinates and charges.

    Args:
        coordinates (dict): A dictionary with atom labels and their coordinates.
        charges (dict): Dictionary containing atomic charges.
        f (float, optional): f = 1/(4pi*ε0) = 332.0716 kcal*Å*mol^(-1)*e^(-2)

    Returns:
        float: The Coulomb energy.
    """
    coulomb_energy = 0

    for atom_label_i, coords_i in coordinates.items():
        for atom_label_j, coords_j in coordinates.items():
            # Atom types (O or H)
            atom_type_i = atom_label_i.split("_")[1][0]
            atom_type_j = atom_label_j.split("_")[1][0]
            # Molecule number
            mol_nb_i = atom_label_i.split("_")[0][0]
            mol_nb_j = atom_label_j.split("_")[0][0]
            # Calculate interactions only between atoms of different molecules
            if mol_nb_i != mol_nb_j:
                r_ij = np.linalg.norm(coords_i - coords_j)
                charge_i = charges[atom_type_i]
                charge_j = charges[atom_type_j]
                coulomb_energy += charge_i * charge_j / r_ij
    return f * coulomb_energy


def compute_total_energy(coordinates, k_bond, l_0, k_angle, theta_0, 
                         verbose=False, epsilons=None, sigmas=None, charges=None):
    """
    Compute the total energy of a molecular system.

    Args:
        coordinates (dict): A dictionary containing atomic coordinates.
        k_bond (float): Bond spring constant.
        l_0 (float): Equilibrium bond length.
        k_angle (float): Angle spring constant.
        theta_0 (float): Equilibrium bond angle.
        verbose (bool, optional): Print verbose output. Defaults to False.
        epsilons (list, optional): List of epsilon values. Defaults to None.
        sigmas (list, optional): List of sigma values. Defaults to None.
        charges (list, optional): List of atomic charges. Defaults to None.

    Returns:
        float : The total energy of the system.
    """
    
    l_OH, theta = calculate_dist_and_angle(coordinates)
    bond_energy = calculate_bond_energy(l_OH, k_bond, l_0)
    angle_energy = calculate_angle_energy(theta, k_angle, theta_0)
    
    if epsilons is not None and sigmas is not None:
        vdw_energy = calculate_vdw_energy(coordinates, epsilons, sigmas)
    else:
        vdw_energy = 0
    if charges is not None:
        coulomb_energy = calculate_coulomb_energy(coordinates, charges)
    else:
        coulomb_energy = 0
    total_energy = bond_energy + angle_energy + vdw_energy + coulomb_energy
    if verbose:
        print(f"{verbose} energy: {total_energy:.2f} kcal/mol.")
        for i in range(len(theta)):
            print(f"{verbose} l_OH_{i+1}: {l_OH[0]:.4f}, {l_OH[1]:.4f} Angstrom.\n"
                f"{verbose} theta_{i+1}: {theta[i] * 180 / np.pi:.2f} degrees.")
    return total_energy


def calculate_grad_num(coordinates, k_bond, l_0, k_angle, theta_0, delta=1e-6, 
                       epsilons=None, sigmas=None, charges=None):
    """
    Calculate the numerical gradient.

    Args:
        coordinates (dict): Dictionary containing atomic coordinates.
        k_bond (float): Bond spring constant.
        l_0 (float): Equilibrium bond length.
        k_angle (float): Angle spring constant.
        theta_0 (float): Equilibrium bond angle.
        delta (float, optional): Small change for numerical differentiation.
        epsilons (dict, optional): Dictionary with epsilon values for atom pairs.
        sigmas (dict, optional): Dictionary with sigma values for atom pairs.
        charges (dict, optional): Dictionary containing atomic charges.

    Returns:
        dict : Numerical gradient.
    """
    grad = {atom: np.zeros_like(coords) for atom, coords in coordinates.items()}
    
    for atom, coords in coordinates.items():
        if len(coords.shape) == 1:
            for i in range(coords.shape[0]):
                coords_plus = copy.deepcopy(coordinates)
                coords_minus = copy.deepcopy(coordinates)
                coords_plus[atom][i] += delta
                coords_minus[atom][i] -= delta
                energy_plus = compute_total_energy(coords_plus, k_bond, l_0, k_angle, theta_0, 
                                                    epsilons, sigmas, charges)
                energy_minus = compute_total_energy(coords_minus, k_bond, l_0, k_angle, theta_0, 
                                                     epsilons, sigmas, charges)
                grad[atom][i] = (energy_plus - energy_minus) / (2 * delta)
        else:
            for j in range(coords.shape[1]):
                coords_plus = {atom: copy.deepcopy(coords)}
                coords_minus = {atom: copy.deepcopy(coords)}
                coords_plus[atom][i] += delta
                coords_minus[atom][i] -= delta
                energy_plus = compute_total_energy(coords_plus, k_bond, l_0, k_angle, theta_0, 
                                                    epsilons, sigmas, charges)
                energy_minus = compute_total_energy(coords_minus, k_bond, l_0, k_angle, theta_0, 
                                                     epsilons, sigmas, charges)
                grad[atom][i, j] = (energy_plus - energy_minus) / (2 * delta)
    return grad


def calculate_grms(gradient):
    """
    Calculate the root mean square of the gradient.
    
    Args:
        gradient (dict): Dictionary containing the gradient values.
        
    Returns:
        float : The root mean square of the gradient.
    """
    gradient_values = np.array(list(gradient.values()))
    squared_gradients = np.square(gradient_values)
    mean_squared_gradient = np.mean(squared_gradients)
    grms = np.sqrt(mean_squared_gradient)
    return grms


def minimize_energy(coordinates, k_bond, l_0, k_angle, theta_0, 
                    max_iter=1000, delta=1e-6, alpha=0.001, eps_lim=1e-10, 
                    epsilons=None, sigmas=None, charges=None):
    """
    Minimize the energy of a molecular system using gradient descent.

    Args:
        coordinates (dict): Dictionary containing atomic coordinates.
        k_bond (float): Bond spring constant.
        l_0 (float): Equilibrium bond length.
        k_angle (float): Angle spring constant.
        theta_0 (float): Equilibrium bond angle.
        max_iter (int, optional): Maximum number of iterations, default is 1000.
        delta (float, optional): Small change for numerical differentiation, default is 1e-6.
        alpha (float, optional): Learning rate for gradient descent, default is 0.001.
        eps_lim (float, optional): Convergence criterion, default is 1e-4.
        epsilons (dict, optional): Dictionary containing epsilon values for atom pairs.
        sigmas (dict, optional): Dictionary containing sigma values for atom pairs.
        charges (dict, optional): Dictionary containing atomic charges.

    Returns:
        tuple: A tuple containing the minimized coordinates and the final energy.
    """
    new_coords = {atom: copy.deepcopy(coords) for atom, coords in coordinates.items()}
    
    for i in range(max_iter):
        
        energy = compute_total_energy(new_coords, k_bond, l_0, k_angle, theta_0, 
                                      epsilons=epsilons, sigmas=sigmas, charges=charges)
        grad = calculate_grad_num(new_coords, k_bond, l_0, k_angle, theta_0, 
                                  delta, epsilons, sigmas, charges)
        
        if calculate_grms(grad) < eps_lim:
            energy = compute_total_energy(new_coords, k_bond, l_0, k_angle, theta_0, verbose="Final",
                                          epsilons=epsilons, sigmas=sigmas, charges=charges)
            print(f"Converged in {i} steps.")
            break

        for atom, coords in new_coords.items():
            new_coords[atom] -= alpha * grad[atom]

    return new_coords, energy


if __name__ == "__main__":
    filename = sys.argv[1]
    coordinates = read_coordinates(filename)
    total_energy = compute_total_energy(coordinates, k_bond, l_0, k_angle, 
                                        theta_0, verbose="Initial")
    new_coords, new_energy = minimize_energy(coordinates, k_bond, l_0, 
                                             k_angle, theta_0)
    print(f"Energy after minimization: {new_energy} kcal/mol")


