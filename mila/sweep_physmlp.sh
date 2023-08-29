#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/ua95pu8h/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/ua95pu8h/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/ActiveLearningMaterials
wandb agent --count 150 mila-ocp/ActiveLearningMaterials/ua95pu8h  # sweep_id
