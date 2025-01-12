User guide
=================

The best starting point for understanding how FreeGSNKE works and how to use the code is to read the `FreeGS documentation
<https://freegs.readthedocs.io/en/latest/>`_ in its entirety. FreeGSNKE makes use of FreeGS objects and functionality, so understanding the FreeGS library first is essential.

To get started with using FreeGSNKE, first read the `home page <../index.html>`_ information from end to end. Then, read through and experiment with the example notebooks in the ``examples`` directory. The recommended order follows, along with a short overview of what each notebook covers:

- ``equilibrium_examples``: describes core FreeGSNKE objects and how to set up static equilibria.
- ``basic_dynamical_evolution``: an example of a simple evolving equilibrium with default settings.
- ``example_nonlinear_evolution_diverted`` and ``example_nonlinear_evolution_limiter``: more involved examples of evolving diverted and limited plasmas.
- ``machine_config``: demonstrates how to build a custom tokamak in FreeGSNKE.

The notebooks are displayed statically on the following pages, but running them interactively and experimenting with the features is highly recommended.

.. toctree::
    :maxdepth: 1
    
    ../notebooks/example0 - build_tokamak_machine
    ../notebooks/example1 - static_inverse_solve_MASTU
    ../notebooks/example2 - static_forward_solve_MASTU.ipynb
    ../notebooks/example3 - extracting_equilibrium_quantites
    ../notebooks/example4 - using_magnetic_probes
    ../notebooks/example5 - evolutive_forward_solve
    ../notebooks/example7 - static_inverse_solve_SPARC
    ../notebooks/example8 - static_inverse_solve_ITER