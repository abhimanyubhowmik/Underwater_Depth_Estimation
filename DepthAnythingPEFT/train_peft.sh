#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output=DepthAnything/scripts/depthAnthingPEFT.out
#SBATCH --error=DepthAnything/scripts/depthAnthingPEFT.err
#SBATCH --partition=mundus,all
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=10:30:00

python depthest/Underwater_Depth_Estimation/DepthAnythingPEFT/run_training.py