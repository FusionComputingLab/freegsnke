# Documentation guide

The documentation is provided by SPHINX and is partially automatically generated from the code. The configuration of the webpage can be changed in `conf.py`. This includes: title, copyright, authors, release, extensions, theme, css, etc..  


## API documentation

The requirements for building the documentation kept separate from the FreeGSNKE requirements. To build the documentation, you also have to have inttalled FreeGSNKE and it's dependencies. To install the documentation requirements, run the following command from the docs directory.

```bash
pip install -r requirements_docs.txt
```

The documentation can then be built by running:

```bash
bash build_documentation.sh
```

## Viewing the documentation

After building you can look at `docs/_build/html/index.html` in the browser to view the documentation webpage.

## Manual documentation

Additionally you can add documentation manually. For this see the examples under `docs/installation_guide/index.rst`, `docs/user_guide/index.rst`, and `docs/user_guide/guide_1.rst`

## TODOs
- [ ] add general documentation and examples/guides
- [ ] update `conf.py` file with correct details (authors, copyright, etc.)
- [ ] Exchange the sphinx-build with [sphinx-multiversion](https://holzhaus.github.io/sphinx-multiversion/master/index.html). (and set it up so that we have different tagged commits that show up as versions on the website)
- [ ] Make sure tests fail if sphinx-build fails
- [ ] add tests to check if documentation is build properly