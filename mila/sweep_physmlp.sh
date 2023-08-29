#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=main
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/s69ywif6/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/s69ywif6/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/run-repos/ActiveLearningMaterials-dev
wandb agent --count 150 mila-ocp/Dave-MBform/s69ywif6
