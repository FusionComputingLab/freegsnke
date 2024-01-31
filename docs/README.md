# Documentation guide

The documentation is provided by Sphinx and is partially automatically generated from the code.


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

## Viewing the documentation

After building, open the `docs/_build/html/index.html` file in a browser to view the documentation landing page.

## Manual documentation

Additionally, documentation can be added manually. See, for example, `docs/user_guide/index.rst`.

## TODOs
- [ ] add general documentation and examples/guides
- [ ] update `conf.py` file with correct details (authors, copyright, etc.)
- [ ] Exchange the sphinx-build with [sphinx-multiversion](https://holzhaus.github.io/sphinx-multiversion/master/index.html). (and set it up so that we have different tagged commits that show up as versions on the website)
- [ ] Make sure tests fail if sphinx-build fails
- [ ] add tests to check if documentation is build properly