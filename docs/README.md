# Documentation guide

The documentation is built using Sphinx and is partially automatically generated from the code.

The documentation is available at [docs.freegsnke.com](https://docs.freegsnke.com/).


## Building the documentation

To build the documentation locally, first install FreeGSNKE with the `docs` extra. For example, the following command may be run from the FreeGSNKE root directory:

```bash
pip install ".[docs]"
```

Other extras can be installed at the same time, for example:

```bash
pip install -e ".[freegs4e,dev,docs]"
```

The documentation can then be built by running the following command from the `docs/` directory:

```bash
bash build_documentation.sh
```

This may take several minutes to complete as some examples of the code are run to generate the documentation.

## Viewing the documentation

After building, open the `docs/_build/html/index.html` file in a browser to view the documentation landing page.

## Developing the documentation

To develop the documentation, you can use the following command to automatically rebuild the documentation when changes are made:

```bash
bash build_documentation.sh live
```

The documentation can then be viewed in a browser at `http://localhost:8000`.
