set -e

make clean

echo "Copying notebook files"
mkdir -p notebooks
cp "../examples/example0 - build_tokamak_machine.ipynb" notebooks
cp "../examples/example1 - static_inverse_solve_MASTU.ipynb" notebooks
cp "../examples/example2 - static_forward_solve_MASTU.ipynb" notebooks
cp "../examples/example3 - extracting_equilibrium_quantites.ipynb" notebooks
cp "../examples/example4 - using_magnetic_probes.ipynb" notebooks
cp "../examples/example5 - evolutive_forward_solve.ipynb" notebooks
cp "../examples/example7 - static_inverse_solve_SPARC.ipynb" notebooks
cp "../examples/example8 - static_inverse_solve_ITER.ipynb" notebooks
cp "../examples/example9 - virtual_circuits_MASTU.ipynb" notebooks
cp ../examples/limiter_currents.json notebooks
cp ../examples/simple_diverted_currents_PaxisIp.pk notebooks
cp ../examples/simple_limited_currents_PaxisIp.pk notebooks

echo "Copying images"
mkdir -p _images
cp ../_images/* _images

echo "Copying machine configurations"
mkdir -p machine_configs
cp -r ../machine_configs/MAST-U machine_configs
cp -r ../machine_configs/SPARC machine_configs
cp -r ../machine_configs/ITER machine_configs

echo "Generating API documentation"
sphinx-apidoc -e -f --no-toc -o api/ ../freegsnke/

echo "Building documentation"
if [ "$1" == "live" ]; then
    sphinx-autobuild . _build/html
else
    make html
fi