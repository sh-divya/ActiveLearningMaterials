#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sweep-physmlp_mbform
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --output=/network/scratch/s/schmidtv/dave/logs/sweeps/output-%j.out
#SBATCH --error=/network/scratch/s/schmidtv/dave/logs/sweeps/error-%j.out

module load anaconda/3 cuda/11.7
conda activate crystal-proxy
cp -r /home/mila/s/schmidtv/ocp-project/ActiveLearningMaterials $SLURM_TMPDIR
cd $SLURM_TMPDIR/ActiveLearningMaterials
wandb agent --count 75 mila-ocp/Dave-MBform/1vvgrwkt  # sweep_id
