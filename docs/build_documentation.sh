set -e

make clean

echo "Copying notebook files"
mkdir -p notebooks
cp "../examples/example00 - build_tokamak_machine.ipynb" notebooks
cp "../examples/example01a - static_inverse_solve_MASTU.ipynb" notebooks
cp "../examples/example01b - advanced_static_inverse_solve.ipynb" notebooks
cp "../examples/example02 - static_forward_solve_MASTU.ipynb" notebooks
cp "../examples/example03 - extracting_equilibrium_quantites.ipynb" notebooks
cp "../examples/example04 - using_magnetic_probes.ipynb" notebooks
cp "../examples/example05 - evolutive_forward_solve.ipynb" notebooks
cp "../examples/example07 - static_inverse_solve_SPARC.ipynb" notebooks
cp "../examples/example08 - static_inverse_solve_ITER.ipynb" notebooks
cp "../examples/example09 - virtual_circuits_MASTU.ipynb" notebooks
cp "../examples/example10 - growth_rates.ipynb" notebooks
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