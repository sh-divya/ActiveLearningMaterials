#!/bin/bash
#SBATCH -J Sweep
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output='/network/scratch/a/alexandre.duval/ocp/runs/sweep/output-%j.out'

module load anaconda/3
conda activate crystals
wandb agent --count 20 mila-ocp/ActiveLearningMaterials/couy0jvg  # sweep_id