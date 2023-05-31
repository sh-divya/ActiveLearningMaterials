#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output='/network/scratch/a/alexandre.duval/ocp/runs/sweep/output-%j.out'
#SBATCH --error='/network/scratch/a/alexandre.duval/ocp/runs/sweep/error-%j.out'

module load anaconda/3
conda activate crystals
cd ~/ocp/ActiveLearningMaterials
wandb agent --count 40 mila-ocp/ActiveLearningMaterials/z96v9jhp  # sweep_id