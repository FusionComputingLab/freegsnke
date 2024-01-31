# FreeGSNKE: Free-boundary Grad-Shafranov Newton-Krylov Evolve

FreeGSNKE (pronounced "free-gee-snake") is an evolutive tokamak plasma
equilibrium simulator. It builds on
[FreeGS](https://github.com/bendudson/freegs) and adds new capabilities. In
particular, FreeGSNKE includes:
- Temporal evolution of plasma equilibria using the linearised circuit equation
  with implicit Euler solver.
- A Newton-Krylov method to replace the Picard method in the Grad-Shafranov
  solver.

It is recommended to read this page in its entirety before attempting to install
or run FreeGSNKE.

## Installation
Building from source is currently the only supported installation method and
requires a working conda installation (for example, through
[Miniforge](https://github.com/conda-forge/miniforge)).

1. Clone this repository and `cd` into the created directory.
2. Set up the conda environment with the command `conda env create -f
   environment.yml`. This creates a conda environment called `freegsnke` with
   the dependencies of FreeGSFast and FreeGSNKE.
> **Note** If this environment setup step fails, try installing the dependencies
> with less restrictive version requirements. 
3. Activate this environment with `conda activate freegsnke`.
4. Clone [FreeGSFast](https://github.com/farscape-project/freegsfast) into a
   directory that is *not* within the FreeGSNKE repository and `cd` into it.
5. Install FreeGSFast locally with `pip install .`.
> **Note** You don't need to follow the installation instructions in the
> FreeGSFast installation directory when installing FreeGSFast as a FreeGSNKE
> dependency. Simply run `pip install .` from the FreeGSFast repository with the
> `freegsnke` conda environment active to install it. Furthermore, the
> FreeGSFast dependencies have already been installed in step 2 so do not need
> to be installed again.
6. `cd` back to the `freegsnke` repository directory and install this package by
   running `pip install .` from the repository home directory. If you plan on
   doing development, it is better to install with symlinks: `pip install -e .`.

## Usage

As FreeGSNKE relies on some core functionality of FreeGS, it is strongly
recommended to first fimiliarise with that code first. Documentation for FreeGS
is available [here](https://freegs.readthedocs.io/en/latest/).

There are two primary sources of information for using FreeGSNKE: the user
documentation and notebook examples.

### Documentation

FreeGSNKE documentation is available using
[Sphinx](https://www.sphinx-doc.org/en/master/). To build the documentation,
execute the following commands:
```bash
cd docs
pip install -r requirements_docs.txt
sphinx-apidoc -e -f --no-toc -o ./api/ ../freegsnke/ 
sphinx-build -b html ./ ./_build/html
```

Open the resulting `index.html` file in a web browser to view the documentation.

### Examples

Jupyter notebooks with example usage are in the `examples/` directory. They
require the `notebook` Python package, which is not listed as a requirement of
FreeGSNKE so must be installed separately. The README in the `examples/`
directory has more information on the examples.

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

Development dependencies are located in `requirements-dev.txt` and can be
installed into the `freegsnke` conda environment with `python -m pip install -r
requirements-dev.txt`.

Several tests are implemented with [pytest](https://docs.pytest.org/en), which
are run as part of the GitLab CI/CD pipelines, but you can run these locally
before submitting a merge request to see if they're likely to pass.

If your bug fix or feature addition includes a change to how FreeGSNKE
fundamentally works or a change to the API, be sure to document this
appropriately in the user documentation, API documentation, and by writing or
changing examples, where appropriate.


## Reference

A paper describing FreeGSNKE is currently in review.