set -e

make clean

echo "Copying notebook files"
mkdir -p notebooks
cp ../examples/example*.ipynb notebooks
cp -r ../examples/data notebooks

echo "Copying images"
mkdir -p _images
cp ../_images/* _images

echo "Copying machine configurations"
mkdir -p machine_configs
cp -r ../machine_configs/* machine_configs

echo "Generating API documentation"
sphinx-apidoc -e -f --no-toc -o api/ ../freegsnke/

echo "Building documentation"
if [ "$1" == "live" ]; then
    sphinx-autobuild . _build/html
else
    make html
fi