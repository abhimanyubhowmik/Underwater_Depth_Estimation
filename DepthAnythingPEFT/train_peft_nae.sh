#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output=depthAnthingPEFT.out
#SBATCH --error=depthAnthingPEFT.err
#SBATCH --partition=mundus,all
#SBATCH --gres=gpu
#SBATCH --time=07:00:00

python /home/mundus/konthuam709/depth_estimation/Underwater_Depth_Estimation/DepthAnythingPEFT/run_training.py