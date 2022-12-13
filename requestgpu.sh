#!/bin/bash
#SBATCH --account=def-rnoumeir
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=24  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=12:00:00     # DD-HH:MM:SS

module restore tensorenvironment
SOURCEDIR=~/scratch/Pointnet
# Prepare virtualenv
source ~/Workspace/TensorFlowEnvironment/bin/activate
tensorboard --logdir=./logs --host 0.0.0.0 --load_fast false & python $SOURCEDIR/train.py
