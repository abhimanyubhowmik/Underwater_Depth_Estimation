#!/bin/bash

#SBATCH --job-name=depthAnthingPEFT
#SBATCH --output= train_log_may/depthAnthingPEFT.out
#SBATCH --error= train_log_may/depthAnthingPEFT.err
#SBATCH --partition=mundus,all
#SBATCH --gres=gpu:a100-20:1
#SBATCH --time=07:00:00

python /home/mundus/konthuam709/depth_estimation/Underwater_Depth_Estimation/DepthAnythingPEFT/run_training_epochs.py