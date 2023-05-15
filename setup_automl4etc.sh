#!/bin/bash
eval "$(conda shell.bash hook)"
conda create --prefix ./venv python=3.8 -y
conda activate ./venv
mv hyperkeras ./venv/lib/python3.8/site-packages/

conda install -y nb_conda_kernels

yes | pip install -r requirements_feb5_2023.txt
yes | pip --default-timeout=1000 install tensorflow==2.11.0
mv hypernets ./venv/lib/python3.8/site-packages/
conda install -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
