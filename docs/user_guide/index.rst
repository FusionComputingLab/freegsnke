User guide
=================

The best starting point for understanding how FreeGSNKE works and how to use the code is to read the `FreeGS documentation
<https://freegs.readthedocs.io/en/latest/>`_ in its entirety. FreeGSNKE makes use of FreeGS objects and functionality, so understanding the FreeGS library first is essential.

The best way of getting started with using FreeGSNKE is to read through and experiment with the example notebooks in the ``examples`` directory. The recommended order follows, along with a short overview of what each notebook covers:

- ``equilibrium_examples``: describes core FreeGSNKE objects and how to set up static equilibria.
- ``basic_dynamical_evolution``: an example of a simple evolving equilibrium with default settings.
- ``example_nonlinear_evolution_diverted`` and ``example_nonlinear_evolution_limiter``: more involved examples of evolving diverted and limited plasmas.
- ``machine_config``: demonstrates how to build a custom tokamak in FreeGSNKE.

The notebooks are displayed statically on the following pages, but running them interactively and experimenting with the features is highly recommended.

.. toctree::
    :maxdepth: 1
    
    ../notebooks/equilibrium_examples
    ../notebooks/basic_dynamical_evolution
    ../notebooks/example_nonlinear_evolution_diverted
    ../notebooks/example_nonlinear_evolution_limiter
    ../notebooks/machine_config