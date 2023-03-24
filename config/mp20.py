import os.path as osp


base_config = {
    "root": "proxies/mp20",
    "input_len": 85,
    "hidden_layers": [[512, 512]],
    "lr": [1e-2, 1e-3, 1e-4, 1e-5],
    "batch_size": [64],
}
scalex = {
    "mean": osp.join(base_config["root"], "x.mean"),
    "std": osp.join(base_config["root"], "x.std"),
}
scaley = {
    "mean": osp.join(config["root"], "y.mean"),
    "std": osp.join(config["root"], "y.std"),
}

config = {
    "project": "MP-20",
    "model_config": base_config,
    "xscale": scalex,
    "yscale": scaley
}

# tuning_parameter = base_config[]

