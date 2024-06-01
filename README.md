This repository includes models and datasets to train and use as proxies as part of a GFlowNet pipline. The aim is to construct/sample crystal objects with a probability that is porportioanl to the reward calculated from the output of the proxy model. Possible targets include 'formation energy per atom' or 'ionic conductivity',

You can look at training results [here](https://wandb.ai/crystal-gfns?shareProfileType=copy)

## Requirements

The code runs on python=3.9, and the required packages can be installed using

```bash
pip install -r requirements_materials.txt
```

## Ionic Conductivity

Dataset can downloaded using functions in utils/mp.py
The model can then be trained using utils/ic\_run.py
The dataset class in utils/crystal\_data.py is being modified
to match cdvae pipleline.

## CDVAE Baseline Datasets

The CDVAE [paper](https://arxiv.org/abs/2110.06197) and repository provides 3 datasets that we will use as baselines for GFlowNet training. The dataset code is under proxies/data.py and can be trained using run.py.

The 'config' folder will contain configurations/hyperparameter dictionaries to search over or use and train.

## Wandb sweeps

1. **Create a yaml file** (’sweep_wandb.yml’) following the instructions given in https://docs.wandb.ai/guides/sweeps/configuration. It contains the parameters we shall sweep over. 
2. **Initiate a wandb sweep** (manually from terminal) with the command: 
`wandb sweep path_to_file/sweep_wandb.yml --name=’test’`. Store the **sweep_id**
3. **Launch a sweep agent** using a slurm script with
`sbatch sweep_mlp.sh` which contains `wandb agent --count 5 mila-ocp/ocp/sweep_id`. The count specificies the number of hyperparam settings to test. To launch several agents (i.e. gpus), use `sbtach --array=0-5`.
4. **Visualise** the results in sweeps section of wandb, under the ActiveLearningMaterials repo. 

## SpaceGroup Encoder

Install [this repo](https://github.com/alexhernandezgarcia/crystallograpy) as a python package.