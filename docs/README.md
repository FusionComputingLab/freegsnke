# Documentation guide

The documentation is provided by SPHINX and is partially automatically generated from the code. The configuration of the webpage can be changed in `conf.py`. This includes: title, copyright, authors, release, extensions, theme, css, etc..  


## API documentation

The requirements for building the documentation kept separate from the FreeGSNKE requirements. To install the documentation requirements, run the following command from the docs directory.

```bash
pip install -r requirements_docs.txt
```

The documentation can then be built by running:

```bash
bash build_documentation.sh
```

## Viewing the documentation

After building you can look at `docs/_build/html/index.html` in the browser to view the documentation webpage. 

- [ ] TODO add how to start a server to view the webpage

## Manual documentation

Additionally you can add documentation manually. For this see the examples under `docs/installation_guide/index.rst`, `docs/user_guide/index.rst`, and `docs/user_guide/guide_1.rst`

## TODOs

- [ ] add general documentation and examples/guides
- [ ] update `conf.py` file with correct details (authors, copyright, etc.)
- [ ] Use CI to automatically create and publish documentation to a (private) server
- [ ] Update theme to be prettier