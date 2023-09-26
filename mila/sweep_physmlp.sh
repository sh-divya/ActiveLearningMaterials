#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/osfcmbqt/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/osfcmbqt/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/run-repos/ActiveLearningMaterials-dev
wandb agent --count 50 mila-ocp/Dave-MBform/osfcmbqt
