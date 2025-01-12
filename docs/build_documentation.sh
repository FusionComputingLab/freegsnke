set -e

make clean

mkdir -p notebooks
# cp "../examples/example0 - build_tokamak_machine.ipynb" notebooks
cp "../examples/example1 - static_inverse_solve_MASTU.ipynb" notebooks
# cp "../examples/example2 - static_forward_solve_MASTU.ipynb" notebooks
# cp "../examples/example3 - extracting_equilibrium_quantites.ipynb" notebooks
# cp "../examples/example4 - using_magnetic_probes.ipynb" notebooks
# cp "../examples/example5 - evolutive_forward_solve.ipynb" notebooks
# cp "../examples/example6a - static_forward_MASTU_solves_from_magnetics_only_data.ipynb" notebooks
# cp "../examples/example6b - static_forward_MASTU_solves_from_magnetics_and_MSE_data.ipynb" notebooks
# cp "../examples/example7 - static_inverse_solve_SPARC.ipynb" notebooks
# cp "../examples/example8 - static_inverse_solve_ITER.ipynb" notebooks

mkdir -p _images
cp ../_images/* _images

mkdir -p machine_configs
cp -r ../machine_configs/MAST-U machine_configs

sphinx-apidoc -e -f --no-toc -o api/ ../freegsnke/

if [ "$1" == "live" ]; then
    sphinx-autobuild . _build/html
else
    make html
fi