set -e

make clean
mkdir notebooks
cp ../examples/*.ipynb notebooks
sphinx-apidoc -e -f --no-toc -o api/ ../freegsnke/ 
make html