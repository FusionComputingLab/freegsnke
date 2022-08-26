# FreeGS Newton-Krylov Evolve
This package extends [FreeGS](https://github.com/bendudson/freegs) by adding two new capabilities:
- Temporal evolution using a linearised circuit equation with implicit Euler solver.
- Newton-Krylov method to replace the Picard method in the Grad-Shafranov solver.

Currently, only the MAST-U tokamak configuration is supported.

The present implementation has possibly inaccurate values for coil resistance and plasma resistivity.

In addition to linearisation of the circuit equation, time derivatives with respect to anything but the currents are neglected (these terms can be re-introduced later if an emulator of FreeGS becomes available).

## Installation
Building from source is currently the only supported installation method.

1. Clone this repository.
2. Create a Python virtual environment or Conda environment and activate it.
> **Note**
> Currently only Python 3.8 has been tested.
3. Install the dependencies listed in `requirements.txt`. This can be done quickly with `pip install -r requirements.txt`.  
> **Note**
> `conda install --file requirements.txt` will not work as FreeGS is only hosted on PyPI.
4. Install this package by running `pip install .` from the respository home directory. If you plan on doing development, it is better to install with symlinks: `pip install -e .`.

## Usage
See the Jupyter notebook in the `example/` directory. This requires the notebook package, which is not listed as a requirement of this package.

## Release notes
### 0.2: 26 August 2022
Included a simple model of the wall (resistance values to be tailored). Re-incorporated the Ldot terms in the circuit equation, missing in previous version. These are solved for with a Newton method (over all currents, active and passive). Timestepping necessary for convergence on Ldot is decoupled from the timestepping necessary to integrate the circuit equations (passive structures may require shorter timescales).
