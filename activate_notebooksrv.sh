#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ./venv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

jupyter notebook
