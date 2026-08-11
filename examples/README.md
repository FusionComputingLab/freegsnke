# Examples

These example Jupyter notebooks are intended to be the **first port of call for new users** trying to get up to speed with the basics of simulating Grad-Shafranov equilibria using FreeGSNKE (the numbering represents approximately increasing complexity!). Most examples will use a MAST-U-like tokamak machine, unless otherwise specified.

| Example notebook | Purpose | Who can use it? |
| ------ | ------ | ------ |
| Example 00 | Learn how to build a generic tokamak object. | Anyone |
| Example 01a | Learn how to use the static inverse solver. | Anyone |
| Example 01b | Learn how to use more advanced features of the static inverse solver. | Anyone |
| Example 02 | Learn how to use the static forward solver. | Anyone |
| Example 03 | Learn how to extract results from a calculated equilibrium. | Anyone |
| Example 04 | Learn how to use the magnetic probes object. | Anyone |
| Example 05a | Learn how to use the nonlinear and linear (with GS) evolutive solver to simulate time-dependent equilibria. | Anyone |
| Example 05b | Learn how to use the linear evolutive solver without solving GS at each timestep. | Anyone |
| Example 05c | Learn how to use the linear evolutive solver (with or without solving GS) with automatic relinearisation enabled. | Anyone |
| Example 06a/b | Simulate (static) MAST-U equilibria over an entire shot using inputs from EFIT++ (requires internal UKAEA MAST-U database). | UKAEA employees + collaborators |
| Example 06c | Audit and prepare the series-connected MAST-U machine for exactly up-down symmetric calculations, including odd-circuit detection, active/passive magnetic-fingerprint changes, supporting geometry discrepancies, and explicit user acceptance (requires access to UDA). | UKAEA employees + collaborators |
| Example 06d | Step through up-down machine symmetrisation using a self-contained MAST-U-like description, including pairing, odd-circuit handling, acceptance diagnostics, current transforms, and plots. | Anyone |
| Example 07a | Static inverse solve in the "Anamak" toy tokamak. | Anyone |
| Example 07b | Static inverse solve in a SPARC-like tokamak. | Anyone |
| Example 07c | Static inverse solve in an ITER-like tokamak. | Anyone |
| Example 09 | Learn how to use and build virtual circuits for plasma shape control. | Anyone |
| Example 10 | Learn how to calculate growth rates associated with vertically unstable modes. | Anyone |
| Example 11 | Learn how to use the evolutive solver alongside a virtual plasma control system (FreeGSNKE Pulse Design Tool). | Anyone |

If a new example has been created, please add a new line to the table explaining its purpose!
