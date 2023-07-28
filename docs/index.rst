FreeGSNKE: Free-boundary Grad-Shafranov Newton-Krylov Evolve
============================================================

This package extends `FreeGS <https://github.com/bendudson/freegs>`__ by
adding two new capabilities: - Temporal evolution using a linearised
circuit equation with implicit Euler solver. - Newton-Krylov method to
replace the Picard method in the Grad-Shafranov solver.

Currently, only the MAST-U tokamak configuration is supported.

The present implementation has possibly inaccurate values for coil
resistance and plasma resistivity.



.. toctree::
    :maxdepth: 2
    :caption: FreeGSNKE
    
    installation_guide/index
    user_guide/index
    api/freegsnke