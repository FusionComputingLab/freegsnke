#!/bin/bash

# $ bash -l build_env_dev.sh
# $ conda activate freegsnke-dev

set -e

ENV_NAME=freegsnke-dev

conda remove --name $ENV_NAME --all -y
conda create --name $ENV_NAME python=3.8 -y
conda activate $ENV_NAME
conda install -c conda-forge jupyter -y
pip install -r requirements.txt
pip install -e .