#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --output='/network/scratch/a/alexandre.duval/ocp/runs/sweep/output-%j.out'

module load-crystals
wandb agent --count 50 mila-ocp/ActiveLearningMaterials/xlzoc91g  # sweep_id