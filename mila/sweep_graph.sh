#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/97g2sar3/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/97g2sar3/error-%j.out

# wandb sweep --entity mila-ocp --project Dave-MBform ./config/sweep/sweep_wandb_graph.yaml

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/run-repos/ActiveLearningMaterials-dev
wandb agent --count 150 mila-ocp/Dave-MBform/97g2sar3
