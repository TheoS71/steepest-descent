# Steepest Descent Algorithm for Water Molecules Energy Minimization

This Python application minimizes the energy of water molecules using the steepest descent algorithm. Two methods of gradient calculation are available: analytical gradient and numerical gradient. There are two scripts available: mini1.py and mini2.py according to the method of gradient calculation and the number of water molecules in the system. Use mini1.py for a single water molecule and mini2.py for any number of water molecules.

### Implementation

To test the algorithm of ONE water molecule, you can use the following command:

```bash
python3 mini1.py <pdbfile> [analytical|numerical]
```

Where `<pdbfile>` is the path to the PDB file containing the coordinates of the water molecule. The analytical method is used by default if no method is specified.

To test the algorithm of ANY number of water molecules, you can use the following command:

```bash
python3 mini2.py <filename> 
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

Where `<filename>` is the path to the file containing the coordinates of the water molecules. The file should be in the following format:

> <molecule> <atom> <atom_number> <x> <y> <z> 
> 1SOL     OW    1   1.713   1.641   1.519
> 1SOL    HW1    2   1.768   1.719   1.520
> 1SOL    HW2    3   1.776   1.568   1.514
