#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --mem=32GB
#SBATCH --output=/network/scratch/d/divya.sharma/dave/sweep/output-%j.out
#SBATCH --error=/network/scratch/d/divya.sharma/dave/sweep/error-%j.out

module load python/3.9
source $HOME/venvs/materials/bin/activate
cp -r $SCRATCH/ActiveLearningMaterials $SLURM_TMPDIR
cd $SLURM_TMPDIR/ActiveLearningMaterials
wandb agent --count 40 mila-ocp/Dave-MBform/mxw7fy6l  # sweep_id
