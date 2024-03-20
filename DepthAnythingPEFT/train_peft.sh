#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output=DepthAnything/scripts/depthAnthingPEFT.out
#SBATCH --error=DepthAnything/scripts/depthAnthingPEFT.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:30:00

python DepthAnything/scripts/run_training.py