#!/bin/bash

conda create --prefix ./venv python=3.8
conda activate ./venv
mv hypernets ./venv/lib/python3.8/site-packages/
mv hyperkeras ./venv/lib/python3.8/site-packages/

conda install nb_conda_kernels

pip install -r requirements_feb5_2023.txt
pip install tensorflow==2.11.0
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
