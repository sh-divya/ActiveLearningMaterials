import os.path as osp
import torch

base_config = {
    "root": "proxies/mp20",
    "input_len": 96,
    "hidden_layers": [512, 512],
    "lr": 1e-2,
    # "lr": [1e-2, 1e-3, 1e-4, 1e-5],
    "batch_size": [32, 64, 128, 256],
}
scalex = {
    "mean": torch.load(osp.join(base_config["root"], "x.mean")),
    "std": torch.load(osp.join(base_config["root"], "x.std")),
}
scaley = {
    "mean": torch.load(osp.join(base_config["root"], "y.mean")),
    "std": torch.load(osp.join(base_config["root"], "y.std")),
}

config = {
    "project": "MP-20",
    "model_config": base_config,
    "xscale": scalex,
    "yscale": scaley,
    "tune_var": "batch_size",
    "target": "formation_energy_per_atom",
    "epochs": 4,
}

# layer_search = [
# [256, 256]
# [128, 128],
# [128, 256, 128],
# [256, 256, 256],
# [128, 256, 256, 128]
# ]
