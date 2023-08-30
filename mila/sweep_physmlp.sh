#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/b9xq7rhh/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/b9xq7rhh/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/run-repos/ActiveLearningMaterials-dev
wandb agent --count 150 mila-ocp/Dave-MBform/b9xq7rhh
