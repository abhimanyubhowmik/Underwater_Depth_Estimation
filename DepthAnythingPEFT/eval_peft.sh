#!/bin/bash

#SBATCH --job-name=evalPEFT
#SBATCH --output=eval/debugged/20_percent/eval.out
#SBATCH --error=eval/debugged/20_percent/eval.err
#SBATCH --partition=mundus,all
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=10:30:00

python /home/mundus/konthuam709/depth_estimation/Underwater_Depth_Estimation/DepthAnythingPEFT/run_evaluation.py