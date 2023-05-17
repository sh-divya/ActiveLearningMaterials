#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --output='/network/scratch/a/alexandre.duval/ocp/runs/sweep/output-%j.out'
#SBATCH --error='/network/scratch/a/alexandre.duval/ocp/runs/sweep/error-%j.out'
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=long

module load anaconda/3
conda activate crystals
cd ~/ocp/ActiveLearningMaterials
python run.py --config graph-mp20