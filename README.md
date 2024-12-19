
<div align="center">
  <img src="images/freegsnke_logo.png" alt="FreeGSNKE Logo" width="200"><br><br>
</div>

# FreeGSNKE: Free-boundary Grad-Shafranov Newton-Krylov Evolve


FreeGSNKE (pronounced "free-gee-snake") is a **Python**-based code for **simulating the evolution of free-boundary tokamak plasma equilibria**.

Based on the well-established [FreeGS](https://github.com/bendudson/freegs) code, it utilises [FreeGS4E](https://github.com/freegs4e/freegs4e) (a fork of FreeGS) to solve different types of free-boundary Grad-Shafranov equilibrium problem and contains a number of new capabilities over FreeGS. 

> **_NOTE:_**  We recommended reading this page in its entirety before attempting to install or run FreeGSNKE!

## Capabilities
FreeGSNKE is capable of solving both **static** (time-<u>in</u>dependent) and **evolutive** (time-dependent) **free-boundary equilibrium problems**. For **fixed-boundary** problems we recommend using FreeGS.

FreeGSNKE can solve:

| Problem Type | Objective | Example use cases | 
| --- | --- | --- |
| **Static forward** | **Solve for the plasma equilibrium** using user-defined poloidal field coil currents, passive structure currents, and plasma current density profiles. | Plasma scenario design and shape control. Equilibrium library generation (for emulation). Initial condition generation for evolutive simulations. |
| **Static inverse** | **Estimate poloidal field coil currents** using user-defined constraints (e.g. isoflux and X-point locations) and plasma current density profiles for a desired plasma equilibrium shape. | Plasma scenario design. Optimisation of poloidal field coil or magnetic probe locations. |
| **Evolutive forward** | **Solve simultaneously for the plasma equilibrium, the poloidal field coil (and passive structure) currents, and the total plasma current over time from an initial equilibrium** using user-defined time-dependent poloidal field coil voltages and plasma current density profile parameters. | Full shot simulations. Vertical stability analysis. |

These problems can be solved in a **user-specified tokamak geometry** that can include:

| Tokamak feature | Purpose | Properties | Element in image below | 
| ------ | ------ | ------ | ------ |
| Active poloidal field coils | Can be assigned (voltage-driven) currents that influence plasma shape and position. | Locations, sizes (areas), wirings (series/anti-series), polarities (+1 or -1), resistivities (of coil materials), and number of windings. | Blue rectangles |
| Passive conducting structures  | Can be assigned induced eddy currents that also impact plasma shape and position. In evolutive forward mode, these are solved self-consistently. | Locations, sizes, orientations (if available), and filaments (as passives can be refined if needed). | Dark grey parallelograms |
| Wall and/or limiter contours  | Confines the plasma boundary (for computational purposes). | Locations. | Solid black line |
| Magnetic diagnostic probes  | Can measure the poloidal flux (fluxloops) or the magnetic field strength (pickup coils) at specified locations. | Locations (for both) and orientations (for pickup coils). | Orange diamonds (fluxloops) and brown dots/lines (pickup coils) |

Each problem is solved using **fourth-order accurate finite differences** and a **purpose-built Newton-Krylov method** for additional **stability and convergence** speed (over the Picard iterations used in FreeGS). For equilibria that have plasma current density profiles with steep edge gradients or sheet currents invoke an **adaptive mesh refinement** scheme on the plasma boundary for additional stability. 

<div align="center">
  <img src="images/mastu_eq.png" alt="mastu_eq" width="350"><br><br>
</div>

Above we show an example of a static equilibrium calculated using FreeGSNKE's forward solver with a **MAST-U** machine description. The contours represent the contours of constant poloidal flux and the different tokamak features are plotted in various colours (refer back to table above). 

## Feature roadmap
FreeGSNKE is constantly evolving and so we hope to provide users with more advanced features over time:

**Short term**:
- JAX-ification of the core Newton-Krylov solvers for auto-differentiability.
- integration with the IMAS data formats. 

**Long term**:
- implementation of the current diffusion equation. 
- coupling with transport solvers. 
- coupling with [MOOSE](https://mooseframework.inl.gov/) to quantify electromagnetic loads on tokamak structures during vertical displacement events. 


## Getting started

**Get familiar with FreeGS(NKE)**: given FreeGSNKE relies on some core FreeGS functionality, it is strongly recommended to first familiarise yourself with how it works by taking a looking at the documentation [here](https://freegs.readthedocs.io/en/latest/).

**After installation (below), check out the FreeGSNKE examples**: in addition to the FreeGS docs, Jupyter notebooks with examples of how to solve each of the above equilibrium problems using FreeGSNKE can be found in the `examples/` directory.

**Refer to the documentation**: once you are a bit more familiar with FreeGSNKE, check out the [Sphinx](https://www.sphinx-doc.org/en/master/) code documentation (build instructions can be found in the `docs/README.md` document).

**References**: check out the references at the bottom of this page for even more detailed information about FreeGSNKE and how it is being used in the community!

**Questions**: for questions or queries about the code, first check the examples, then the documentation, then the references, and then the open/closed issues tab. If those sources don't answer your query, please open an issue and use the 'question' label.


## Installation

Building from source is currently the only supported installation method.

### Stage one: set up a Python environment

The recommended way to install FreeGSNKE is to use a virtual environment such as conda or venv. The following instructions will set up a conda environment:

1. Install the latest [Miniforge](https://github.com/conda-forge/miniforge) distribution for your operating system.
2. Create a new conda environment with:
```shell
conda create -n freegsnke python=3.10
```
3. Activate the new environment with:
```shell
conda activate freegsnke
```

### Stage two: install FreeGSNKE

1. Clone the FreeGSNKE repository with:
```shell
git clone git@gitlab.stfc.ac.uk:farscape-ws3/freegsnke.git
```
or 
```shell
git clone https://gitlab.stfc.ac.uk/farscape-ws3/freegsnke.git
```
2. Enter the FreeGSNKE directory with:
```shell
cd
```
3. Install FreeGSNKE and its dependencies with
```shell
pip install ".[freegs4e]"
```

The extra `freegs4e` dependency in the last step installs [FreeGS4E](https://github.com/freegs4e/freegs4e) automatically (and is required for FreeGSNKE to run). 

If you are planning to develop FreeGSNKE, see the below section on [contributing](#contributing) code.


## Contributing

We welcome contributions including **bug fixes** or **new feature** requests for FreeGSNKE. To do this, the first step is to consider opening an issue on the project's homepage.

**If the issue is a bug**:
- Make sure you're using the latest version of the code as the bug might have been squashed in later releases.
- Search the open and closed issues to see if an issue describing the bug already exists.
- If the bug still persists, open a new issue and include the following:
    - a brief overview of the problem.
    - an explanation of the expected behaviour and the observed behaviour.
    - if possible, a minimum working example for reproducibility.
    - if possible, provide details of the culprit and a suggested fix.

**If the issue is a new feature request**:
- Give a brief overview of the desired feature.
- Explain why it would be useful (extra consideration will be given to features that will benefit the broader community).
- If possible, suggest how the new feature could be implemented.

<details closed>
<summary><strong>Click here to see how to contribute code!</strong></summary>

To make code contributions, please do so via **merge request**. This will require working on your own branch, making the desired changes, and then submitting a merge request. The request will then be considered by the repository maintainers. 

To work on your code in development mode, run the following in place of the final step in [installation](#installation):
```shell
pip install -e ".[freegs4e,dev]"
```
from the from your FreeGSNKE root directory to install FreeGSNKE in editable mode, including the optional development dependencies.

If you are also planning to co-develop [FreeGS4E](https://github.com/freegs4e/freegs4e), you will need to install it in editable mode as well. This can be done by cloning the FreeGS4E repository, installing using the development instructions, and then installing FreeGSNKE in editable mode with:
```shell
pip install ".[dev]"
```
Notice that the `freegs4e` extra has been omitted from the FreeGSNKE installation command in this case.

Please also install the pre-commit hooks ([Black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/)) for code formatting. The [pre-commit](https://pre-commit.com/) library is included in `requirements-dev.txt` and will be installed automatically using the `dev` extra included in the commands above. To install the pre-commit hooks, run the following in the root FreeGSNKE directory after installation:
```shell
pre-commit install
```
> **_NOTE:_** Several tests have been built using [pytest](https://docs.pytest.org/en) and are run as part of the GitLab CI/CD pipelines, but you can run these locally before submitting a merge request if you wish. These must pass in order for the merge request to be approved, so please do fix any errors that pop up if you see them. 

> **_NOTE:_**  If your bug fix or feature addition includes a change to how FreeGSNKE fundamentally works or requires a change to the API, be sure to document this appropriately in the user documentation, API documentation, and by writing/changing the notebook examples where appropriate. Also be sure to fully justify why such changes are needed or necessary.

> **_NOTE:_**  Any Jupyter notebooks tracked by the repository should **not** include cell outputs so that we can keep the size of the repository reasonable. Please do clear these manually in the notebook itself before submitting merge requests.

</details>


## References

If you make use of FreeGSNKE, please do cite our work:

```bibtex
@article{amorisco2024,
	title = {{FreeGSNKE}: A Python-based dynamic free-boundary toroidal plasma equilibrium solver},
  author = {Amorisco, N. C. and Agnello, A. and Holt, G. and Mars, M. and Buchanan, J. and Pamela, S.},
	journal = {Physics of Plasmas},
	volume = {31},
	number = {4},
	pages = {042517},
	year = {2024},
  doi = {10.1063/5.0188467},
}
```

Here are a list of FreeGSNKE-related papers: 

| 2024 |  |
| ------ | ------ | 
|  | N. C. Amorisco et al, "FreeGSNKE: A Python-based dynamic free-boundary toroidal plasma equilibrium solver", Physics of Plasmas, **31**, 042517, (2024). DOI: [10.1063/5.0188467](https://doi.org/10.1063/5.0188467). |
|  | A. Agnello et al, "Emulation techniques for scenario and classical control design of tokamak plasmas", Physics of Plasmas, **31**, 043091, (2024). DOI: [10.1063/5.0187822](https://doi.org/10.1063/5.0187822). |
|  | K. Pentland et al, "Validation of the static forward Grad-Shafranov equilibrium solvers in FreeGSNKE and Fiesta using EFIT++ reconstructions from MAST-U", arXiv, (2024). DOI: [2407.12432](https://arxiv.org/abs/2407.12432). |
|  | 

If you would like your FreeGSNKE-related paper to be added, please let us know!


## Funding

This work was funded under the Fusion Computing Lab collaboration between the STFC Hartree Centre and the UK Atomic Energy Authority. 

## License

FreeGSNKE is currently available under academic license. Contact the authors directly for details.