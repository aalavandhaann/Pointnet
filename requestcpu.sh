#!/bin/bash
#SBATCH --account=def-rnoumeir
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=2046M
#SBATCH --time=0:15:0
python dataprocessing.py
