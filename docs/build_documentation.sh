set -e

make clean

mkdir -p notebooks
cp ../examples/*.ipynb notebooks

mkdir -p _images
cp ../_images/* _images

sphinx-apidoc -e -f --no-toc -o api/ ../freegsnke/

if [ "$1" == "live" ]; then
    sphinx-autobuild . _build/html
else
    make html
fi