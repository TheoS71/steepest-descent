# Steepest Descent Algorithm for Water Molecules Energy Minimization

---
Authors:

**ZHUKOVA Nadezhda** - **SERRALTA Théo**

Master 1 Bio-informatics at *Univerité de Paris*.

---

This Python application minimizes the energy of water molecules using the steepest descent algorithm. Two methods of gradient calculation are available: analytical gradient and numerical gradient. There are two scripts available: mini1.py and mini2.py according to the method of gradient calculation and the number of water molecules in the system. Use mini1.py for a single water molecule and mini2.py for any number of water molecules.

### Create the environment

To clone the repository, use the following command:

```bash
git clone git@github.com:zhukovanadezhda/steepest-descent.git
```

To setup the conda environment :

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [mamba](https://github.com/mamba-org/mamba). Create the `steepest-descent` conda environment:

```bash
mamba env create -f binder/environment.yml
cd steepest-descent
conda activate steepest-descent
```

### Run the application

To test the algorithm of ONE water molecule, you can use the following command:

```bash
python3 scripts/mini1.py <pdbfile> [analytical|numerical]
```

Where `<pdbfile>` is the path to the PDB file containing the coordinates of the water molecule. The analytical method is used by default if no method is specified. The file should be in the following format:

```
ATOM      1  OH  OSP3    1       4.013   0.831  -9.083  1.00  0.00              
ATOM      2 1HH  OSP3    1       4.941   0.844  -8.837  1.00  0.00              
ATOM      3 2HH  OSP3    1       3.750  -0.068  -9.293  1.00  0.00
...
```

You can modify various minimization parameters in the main program of the script. Here's what they correspond to:

- `h`: Small value used for numerical gradient calculation.
- `step_size`: Step size for updating coordinates during minimization.
- `threshold`: Convergence threshold for the Gradient Root Mean Square (GRMS). Optimization stops when GRMS is below this threshold.
- `max_iterations`: Maximum number of iterations allowed to avoid infinite loops.
- `k_bond`: Force constant for bond energy calculation.
- `i_eq`: Equilibrium distance for bond lengths.
- `k_angle`: Force constant for angle energy calculation.
- `theta_eq`: Equilibrium angle in degrees.


To test the algorithm of ANY number of water molecules, you can use the following command:

```bash
python3 scripts/mini2.py <filename> 
```

Where `<filename>` is the path to the file containing the coordinates of the water molecules. The file should be in the following format:

```
1SOL     OW    1   1.713   1.641   1.519
1SOL    HW1    2   1.768   1.719   1.520
1SOL    HW2    3   1.776   1.568   1.514
...
```
> #### Attention!
> Don't forget to update water molecule system constants according to the number of molecules. The default mode constants work for 3 molecules.

### Examples of usage

1. One water molecule energy minimisation with analytical gradient

```bash
python3 scripts/mini1.py data/one_water_1.txt analytical
```
```
Nombre maximal d'itérations atteint sans convergence.
Iteration 1000 - GRMS: 1.0966692735016927e-09
Énergie finale : 1.2148634994298813e-19 kcal/mol
Angle après minimisation : 104.5200 degrés
Longueur de liaison l1 après minimisation : 0.9572 Å
Longueur de liaison l2 après minimisation : 0.9572 Å
```
2. One water molecule energy minimisation with numerical gradient

```bash
python3 scripts/mini1.py data/one_water_1.txt numerical
```
```
Convergence atteinte. Arrêt de l'optimisation.
Iteration 130 - GRMS: 9.827304249832638e-11
Énergie finale : 1.7770123251781158e-22 kcal/mol
Angle après minimisation : 104.5200 degrés
Longueur de liaison l1 après minimisation : 0.9572 Å
Longueur de liaison l2 après minimisation : 0.9572 Å
```

3. Three water molecules energy minimisation with numerical gradient
```bash
python3 scripts/mini2.py data/one_water_2.txt
```
```
Initial energy: 333.75 kcal/mol.
Initial l_OH_1: 0.0954, 0.0966 Angstrom.
Initial theta_1: 104.03 degrees.
Final energy: 0.00 kcal/mol.
Final l_OH_1: 0.9572, 0.9572 Angstrom.
Final theta_1: 104.52 degrees.
Converged in 132 steps.
Energy after minimization: 2.02614832830815e-22 kcal/mol
```
