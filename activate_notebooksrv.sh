#!/bin/bash

conda activate ./venv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

jupyter notebook
