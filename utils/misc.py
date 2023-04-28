import copy
import os
import random
from itertools import product
from os.path import expandvars
from pathlib import Path
from typing import Dict, List, Union
from uuid import uuid4

import numpy as np
import torch
from yaml import dump, safe_load

from utils.parser import parse_args_to_dict

JOB_ID = os.environ.get("SLURM_JOB_ID")
ROOT = Path(__file__).resolve().parent.parent  # repo root Path


def resolve(path: Union[str, Path]) -> Path:
    """
    Resolve a path to an absolute ``pathlib.Path``, expanding environment variables and
    user home directory.

    Args:
        path: The path to resolve.

    Returns:
        The resolved path.
    """
    return Path(expandvars(path)).expanduser().resolve()


ROOT = resolve(__file__).parent.parent  # repo root Path


def is_mila() -> bool:
    """
    Check if the code is running on the Mila cluster.

    Returns:
        bool: True if the code is running on the Mila cluster.
    """
    return resolve("~").parts[2] == "mila"


def get_run_dir() -> Path:
    """
    Get the run directory.
    On Mila, it is $SCRATCH/crystals-proxys/runs/SLURM_JOB_ID.
    Otherwise, they are in a random sub-directory of checkpoints/

    If the directory already exists, a new one is created with a suffix "-1", "-2", etc.

    Example:
    .. code-block:: python
        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/crystals-proxys/runs/3037262

        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/crystals-proxys/runs/3037262-1

        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/crystals-proxys/runs/3037262-2

    Returns:
        Path: new (unique) run dir for this execution
    """
    if is_mila() and JOB_ID is not None:
        dirpath = resolve(f"$SCRATCH/crystals-proxys/runs/{JOB_ID}")
    else:
        u = str(uuid4()).split("-")[0]
        dirpath = ROOT / "checkpoints" / u

    if dirpath.exists():
        i = (
            max(
                [
                    float(p.name.split("-")[-1])
                    for p in dirpath.parent.glob(f"{dirpath.name}-*")
                ],
                default=0,
            )
            + 1
        )
        dirpath = dirpath.parent / f"{dirpath.name}-{int(i)}"
    print("Run dir:", (dirpath))
    return dirpath


def print_config(config: dict) -> None:
    """
    Print a config dictionary to stdout.
    Discard the "xscale" and "yscale" keys.

    Args:
        config (dict): config dictionary to print
    """
    config = copy.deepcopy(config)
    for scale, scale_dict in config.get("scales", {}).items():
        if "mean" in scale_dict:
            if isinstance(scale_dict["mean"], torch.Tensor):
                config["scales"][scale]["mean"] = "Tensor with shape " + str(
                    scale_dict["mean"].shape
                )
        if "std" in scale_dict:
            if isinstance(scale_dict["std"], torch.Tensor):
                config["scales"][scale]["std"] = "Tensor with shape " + str(
                    scale_dict["std"].shape
                )
    print()
    print("#" * 50)
    print("#" * 50)
    print()
    print(dump(config))
    print("#" * 50)
    print("#" * 50)
    print()


def flatten_grid_search(grid: Dict[str, List]) -> List[Dict]:
    """
    Creates the cartesian product of all the values in `grid`, and returns a list of
    dictionaries.

    Example:

    .. code-block:: python
        >>> flatten_grid_search({"a": [1, 2], "b": [3, 4]})
        [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]

    Args:
        grid (Dict[str, List]): The grid's parameterization

    Returns:
        List[Dict]: List of all HP combinations
    """
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, v)) for v in product(*values)]


def merge_dicts(dict1: dict, dict2: dict, resolve_lists=None) -> dict:
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary
    as a value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Example:

    .. code-block:: python
        >>> merge_dicts({"a": 1, "b": 2, "c": {"d": 3}}, {"b": 3, "c": {"d": 4}})
        {"a": 1, "b": 3, "c": {"d": 4}}

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share
        the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)} {dict1}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)} {dict2}.")

    return_dict = copy.deepcopy(dict1)

    assert resolve_lists in [None, "overwrite"]

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(
                    dict1[k], dict2[k], resolve_lists=resolve_lists
                )
            elif isinstance(v, list) and isinstance(dict1[k], list):
                if len(dict1[k]) != len(dict2[k]):
                    if resolve_lists == "overwrite":
                        print(
                            f"Overwriting (not merging) list for key {k} because "
                            + "of different list length"
                        )
                        return_dict[k] = v
                        continue
                    raise ValueError(
                        f"List for key {k} has different length in dict1 and dict2."
                        + " Use an empty dict {} to pad for items in the shorter list."
                    )
                if isinstance(dict1[k][0], dict):
                    if not isinstance(dict2[k][0], dict):
                        raise ValueError(
                            f"Expecting dict for key {k} in dict2. ({dict1}, {dict2})"
                        )
                    return_dict[k] = [
                        merge_dicts(d1, d2, resolve_lists=resolve_lists)
                        for d1, d2 in zip(dict1[k], v)
                    ]
                else:
                    if isinstance(dict2[k][0], dict):
                        raise ValueError(
                            f"Expecting dict for key {k} in dict1. ({dict1}, {dict2})"
                        )
                    return_dict[k] = v

            else:
                return_dict[k] = dict2[k]

    return return_dict


def load_scales(config):
    if "scales" not in config:
        return config

    scales = {s: {} for s in config["scales"]}

    for scale, scale_conf in config["scales"].items():
        if "load" in scale_conf:
            src = config["src"].replace("$root", str(ROOT))
            if src.startswith("/"):
                src = resolve(src)
            else:
                src = ROOT / src
            assert "mean" in scale_conf and "std" in scale_conf
            if scale_conf["load"] == "torch":
                scales[scale]["mean"] = torch.load(
                    scale_conf["mean"].replace("$src", str(src))
                )
                scales[scale]["std"] = torch.load(
                    scale_conf["std"].replace("$src", str(src))
                )
        else:
            scales[scale] = scale_conf

    config["scales"] = scales

    return config


def load_config() -> dict:
    # 1. parse command-line args
    cli_conf = parse_args_to_dict()
    assert (
        "config" in cli_conf
    ), "Must specify config string as `--config={task}-{model}`"
    # 2. load config files
    model, task = cli_conf["config"].split("-")
    task_file = ROOT / "config" / "tasks" / f"{task}.yaml"
    model_file = ROOT / "config" / "models" / f"{model}.yaml"
    assert task_file.exists(), f"Task config file {str(task_file)} does not exist."
    assert model_file.exists(), f"Model config file {str(model_file)} does not exist."
    config = merge_dicts(
        safe_load(task_file.read_text()),
        safe_load(model_file.read_text()),
    )
    # 3. merge with command-line args
    config = merge_dicts(config, cli_conf, resolve_lists="overwrite")
    if "run_dir" not in config:
        # 3.0 get run dir path if none is specified
        config["run_dir"] = get_run_dir()

    # 3.1 resolve paths
    config["run_dir"] = resolve(config["run_dir"])
    # 3.2 make run directory
    if not config.get("debug"):
        config["run_dir"].mkdir(parents=True, exist_ok=True)
    config["run_dir"] = str(config["run_dir"])

    if "scales" in config:
        config = load_scales(config)
    return config


def set_seeds(seed=0):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
