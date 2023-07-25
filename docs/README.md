# Documentation guide

The documentation is provided by SPHINX and is partially automatically generated from the code. The configuration of the webpage can be changed in `conf.py`. This includes: title, copyright, authors, release, extensions, theme, css, etc..  


## API documentation
The api documentation is created by running:

```bash
cd docs
bash build_documentation.sh
```
## Viewing the documentation
After this you can look at `docs/index.html` in the browser to view the documentation webpage. 

- [ ] TODO add how to start a server to view the webpage

## Manual documentation
Additionally you can add documentation manually. For this see the examples under `docs/installation_guide/index.rst`, `docs/user_guide/index.rst`, and `docs/user_guide/guide_1.rst`

## TODOs
- [ ] add general documentation and examples/guides
- [ ] update `conf.py` file with correct details (authors, copyright, etc.)
- [ ] Use CI to automatically create and publish documentation to a (private) server
- [ ] Update theme to be prettier