import warnings
from yaml import safe_load
from pathlib import Path
import subprocess
import json
from functools import partial

import optuna

warnings.filterwarnings("ignore", ".*does not have many workers.*")

BASE_PATH = Path(__file__).parent.parent


def read_sweep(yptr):
    fobj = open(BASE_PATH / "config" / "sweep" / f"{yptr}.yaml")
    sweep_cfg = safe_load(fobj)
    dix = {}
    for prm in sweep_cfg["parameters"]:
        dix[prm] = sweep_cfg["parameters"][prm]["values"]
    return dix, sweep_cfg["command"]


def objective(trial, cfg, cmd):
    new_cfg = {}
    for k, v in cfg.items():
        if len(v) > 0:
            new_cfg[k] = trial.suggest_categorical(k, v)
        else:
            new_cfg[k] = v[0]
    for k, v in new_cfg.items():
        cmd = cmd + [f"--{k}={v}"]
    subprocess.call(cmd)
    wandb_run = BASE_PATH / "wandb" / "latest-run" / "files" / "wandb-summary.json"
    return json.load(wandb_run.open(encoding="UTF-8"))["Avg Best MAE"]


def run(sweep_yaml):
    prms, base_cmd = read_sweep(sweep_yaml)
    cmd = ["python", str((BASE_PATH / "run.py").resolve())] + base_cmd[3:-1]
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, cfg=prms, cmd=cmd), n_trials=600, timeout=9000)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run("sweep_mlp_ic")
