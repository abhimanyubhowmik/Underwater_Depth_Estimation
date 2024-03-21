#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output=DepthAnything/scripts/depthAnthingPEFT.out
#SBATCH --error=DepthAnything/scripts/depthAnthingPEFT.err
#SBATCH --partition=all,gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=06:30:00

python DepthAnything/scripts/run_training.py