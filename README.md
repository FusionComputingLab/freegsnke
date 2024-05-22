# FreeGSNKE: Free-boundary Grad-Shafranov Newton-Krylov Evolve

FreeGSNKE (pronounced "free-gee-snake") is an evolutive tokamak plasma
equilibrium simulator. It builds on
[FreeGS](https://github.com/bendudson/freegs) and adds new capabilities. In
particular, FreeGSNKE includes:
- A Newton-Krylov method to replace the Picard method in the Grad-Shafranov
  solver.
- Temporal evolution of plasma equilibria coupled with active coils and passive metal structures.

It is recommended to read this page in its entirety before attempting to install
or run FreeGSNKE.

## Installation
Building from source is currently the only supported installation method and follows these broad steps, which are detailed below:
1. Get access to FreeGSFast.
2. Set up a Python environment.
3. Install FreeGSNKE.

The steps will be significantly simplified in the future when FreeGSFast is made public.

### Get access to FreeGSFast

FreeGSNKE is built on top of [FreeGSFast](https://github.com/farscape-project/freegsfast), which is a fork of FreeGS with some performance improvements. Currently, FreeGSFast is in a private repository on the FARSCAPE GitHub, so some extra configuration is required.

Once you have access to the FreeGSFast repository, add an SSH key to your GitHub account by following the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

### Set up a Python environment

The recommended way to install FreeGSNKE is to use a virtual environment using, for example, conda or venv. Below are instructions for setting up a conda environment.

1. Install the latest [Miniforge](https://github.com/conda-forge/miniforge) distribution for your operating system.
2. Create a new conda environment with the command `conda create -n freegsnke python=3.10`.
3. Activate the new environment with `conda activate freegsnke`.

### Install FreeGSNKE

1. Clone the FreeGSNKE repository with `git clone git@gitlab.stfc.ac.uk:farscape-ws3/freegsnke.git` or `git clone https://gitlab.stfc.ac.uk/farscape-ws3/freegsnke.git`.
2. `cd` into the FreeGSNKE directory.
3. Install FreeGSNKE and its dependencies with `pip install .`.

If you are planning to develop FreeGSNKE, see the below section on contributing
code.

## Usage

As FreeGSNKE relies on some core functionality of FreeGS, it is strongly
recommended to first fimiliarise with that code first. Documentation for FreeGS
is available [here](https://freegs.readthedocs.io/en/latest/).

There are two primary sources of information for using FreeGSNKE: the user
documentation and notebook examples.

### Documentation

FreeGSNKE documentation is available using
[Sphinx](https://www.sphinx-doc.org/en/master/). Documentation build
instructions are in the `docs/README.md` document.

### Examples

Jupyter notebooks with example usage are in the `examples/` directory. The
README in the `examples/` directory has more information on the examples.

### Questions

For questions about operation of the code or similar, first check the
documentation, provided examples, and closed and open issues. If those sources
don't answer your query, please open an issue and use the 'question' label.

## License

FreeGSNKE is currently available under academic license. Contact the authors
directly for details.

## Contributing

We welcome contributions of bug fixed or feature improvements to FreeGSNKE. To
do so, the first step is to consider opening an issue on the project's home on
STFC GitLab.

If the issue is a bug:
- Make sure you're using the latest version of the code as the bug might have
  been squashed in later releases.
- Search the open and closed issues to see if an issue describing the bug
  already exists.
  
If an issue doesn't exist and the bug still occurs on the
latest version of the code, open an issue and populate it with the following:
- Give a brief overview of the problem.
- Explain the expected behaviour and the observed behaviour.
- Provide a minimum working example for reproducibility.
- If possible, provide details of the culprit and a suggested fix.

If the issue is a feature improvement request:
- Give a brief overview of the desired feature.
- Explain why it would be useful. Extra consideration will be given to features
  that will benefit the broader community.
- If possible, suggest how the new feature could be implemented.

### Contributing code

To make code contributions, please do so via merge request.

In place of the final step in [installation](#installation), run `pip install -e
.[dev]` from the FreeGSNKE root directory to install FreeGSNKE in editable mode, including the optional development dependencies.

Several tests are implemented with [pytest](https://docs.pytest.org/en), which
are run as part of the GitLab CI/CD pipelines, but you can run these locally
before submitting a merge request to see if they're likely to pass.

If your bug fix or feature addition includes a change to how FreeGSNKE
fundamentally works or a change to the API, be sure to document this
appropriately in the user documentation, API documentation, and by writing or
changing examples, where appropriate.

[Black](https://github.com/psf/black) and
[isort](https://pycqa.github.io/isort/) are used for code formatting.

Pre-commit hooks are available in `.pre-commit-config.yaml`, including Black and
isort formatting. The [pre-commit](https://pre-commit.com/) library is included
in `requirements-dev.txt`. To install the pre-commit hooks, run `pre-commit
install` from the root FreeGSNKE directory.

Any Jupyter notebooks tracked by the repository should not include cell outputs.
This is to keep the size of the repository reasonable. These can be cleared
manually in the notebook or `nbconvert` can be used (which is also implemented
as pre-commit hook).

## Reference

A paper describing FreeGSNKE is currently in review.