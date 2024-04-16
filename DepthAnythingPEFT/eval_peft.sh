#!/bin/bash

#SBATCH --job-name=evalPEFT
#SBATCH --output=DepthAnything/scripts/eval.out
#SBATCH --error=DepthAnything/scripts/eval.err
#SBATCH --partition=mundus,all
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=10:30:00

python DepthAnything/scripts/run_evaluation.py