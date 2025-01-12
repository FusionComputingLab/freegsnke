# Documentation guide

The documentation is built using Sphinx and is partially automatically generated from the code.


## API documentation

The requirements for building the documentation are kept separate from the FreeGSNKE requirements. FreeGSNKE and its dependencies are required to be installed to build the docs (see the README in the repository root directory for instructions).

To install the documentation requirements, run the following command from the `docs` directory. Ensure that the active programming environment has FreeGSNKE installed.

```bash
pip install -r requirements_docs.txt
```

The documentation can then be built by running:

```bash
bash build_documentation.sh
```

This may take several minutes to complete as it runs some examples of the code to generate the documentation.

## Viewing the documentation

After building, open the `docs/_build/html/index.html` file in a browser to view the documentation landing page.

## Developing the documentation

To develop the documentation, you can use the following command to automatically rebuild the documentation when changes are made:

```bash
bash build_documentation.sh live
```

The documentation can then be viewed in a browser at `http://localhost:8000`.
