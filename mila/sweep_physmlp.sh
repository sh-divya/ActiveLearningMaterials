#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --output=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/1ibghlxg/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/crystals-proxys/runs/sweeps/1ibghlxg/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cd /home/mila/s/schmidtv/ocp-project/run-repos/ActiveLearningMaterials-dev

echo "Current config:"
echo "• Current working directory: $(pwd)"
echo "• Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "• Current commit: $(git rev-parse HEAD)"

wandb agent --count 50 mila-ocp/Dave-MBform/1ibghlxg
