import contextlib
import copy
import os
import random
import re
import subprocess
import sys
from itertools import product
from os.path import expandvars
from pathlib import Path
from typing import Dict, List, Union
from uuid import uuid4

import numpy as np
import torch
from yaml import dump, safe_load

from dave.utils.parser import parse_args_to_dict


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


JOB_ID = os.environ.get("SLURM_JOB_ID")
ROOT = resolve(__file__).parent.parent.parent  # repo root Path


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
    On Mila, it is $SCRATCH/dave/runs/SLURM_JOB_ID.
    Otherwise, they are in a random sub-directory of checkpoints/

    If the directory already exists, a new one is created with a suffix "-1", "-2", etc.

    Example:
    .. code-block:: python
        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/dave/runs/3037262

        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/dave/runs/3037262-1

        d = get_run_dir()
        d.mkdir()
        print(d) # /home/mila/schmidtv/dave/runs/3037262-2

    Returns:
        Path: new (unique) run dir for this execution
    """
    if is_mila() and JOB_ID is not None:
        dirpath = resolve(f"$SCRATCH/dave/runs/{JOB_ID}")
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
    print(dump(config, default_flow_style=None))
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
    cli_conf["cmd"] = " ".join(sys.argv)
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

    config = set_cpus_to_workers(config)
    return config


def set_seeds(seed=0):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def find_ckpt(ckpt_path: dict, release: str) -> Path:
    """
    Finds a checkpoint in a dictionary of paths, based on the current cluster name
    and release.
    If the path is a file, use it directly.
    Otherwise, look for a single checkpoint file in a ${release}/ sub-fodler.
    E.g.:
        ckpt_path = {"mila": "/path/to/ckpt_dir"}
        release = v2.3_graph_phys
        find_ckpt(ckpt_path, release) -> /path/to/ckpt_dir/v2.3_graph_phys/name.ckpt

        ckpt_path = {"mila": "/path/to/ckpt_dir/file.ckpt"}
        release = v2.3_graph_phys
        find_ckpt(ckpt_path, release) -> /path/to/ckpt_dir/file.ckpt

    Args:
        ckpt_path (dict): Where to look for the checkpoints.
            Maps cluster names to paths.

    Raises:
        ValueError: The current location is not in the checkpoint path dict.
        ValueError: The checkpoint path does not exist.
        ValueError: The checkpoint path is a directory and contains no .ckpt file.
        ValueError: The checkpoint path is a directory and contains >1 .ckpt files.

    Returns:
        Path: Path to the checkpoint for that release on this host.
    """
    loc = os.environ.get(
        "SLURM_CLUSTER_NAME", os.environ.get("SLURM_JOB_ID", os.environ["USER"])
    )
    if all(s.isdigit() for s in loc):
        loc = "mila"
    if loc not in ckpt_path:
        raise ValueError(f"DAV proxy checkpoint path not found for location {loc}.")
    path = resolve(ckpt_path[loc])
    if not path.exists():
        raise ValueError(f"DAV proxy checkpoint not found at {str(path)}.")
    if path.is_file():
        return path
    path = path / release
    ckpts = list(path.glob("*.ckpt"))
    if len(ckpts) == 0:
        raise ValueError(f"No DAV proxy checkpoint found at {str(path)}.")
    if len(ckpts) > 1:
        raise ValueError(
            f"Multiple DAV proxy checkpoints found at {str(path)}. "
            "Please specify the checkpoint explicitly."
        )
    return ckpts[0]


def prepare_for_gfn(ckpt_path_dict, release, rescale_outputs, verbose=True):
    """
    Loads a checkpoint and prepares it for use in the GFlowNet.

    Args:
        ckpt_path_dict (dict): Dictionary mapping cluster names to checkpoint paths.
        rescale_outputs (bool): Whether to rescale the inputs and outputs of the model.
            Inputs would be standardized and output would be rescaled to the original
            scale.
        verbose (bool, optional): . Defaults to True.

    Returns:
        _type_: _description_
    """
    from dave.proxies.models import make_model
    from dave.utils.loaders import make_loaders

    if verbose:
        print("  Making model...")
    # load the checkpoint
    ckpt_path = find_ckpt(ckpt_path_dict, release)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # extract config
    model_config = ckpt["hyper_parameters"]
    scales = model_config.get("scales")
    if rescale_outputs:
        assert scales is not None
        assert all(t in scales for t in ["x", "y"])
        assert all(u in scales[t] for t in ["x", "y"] for u in ["mean", "std"])
    # make model from ckpt config
    model = make_model(model_config)
    proxy_loaders = make_loaders(model_config)
    # load state dict and remove potential leading `model.` in the keys
    if verbose:
        print("  Loading proxy checkpoint...")
    model.load_state_dict(
        {
            k[6:] if k.startswith("model.") else k: v
            for k, v in ckpt["state_dict"].items()
        }
    )
    assert hasattr(model, "pred_inp_size")
    model.n_elements = 89  # TEMPORARY for release `v0-dev-embeddings`
    assert hasattr(model, "n_elements")
    model.eval()
    if verbose:
        print("Proxy ready.")

    return model, proxy_loaders, scales


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run_command(command):
    """
    Run a shell command and return the output.
    """
    return subprocess.check_output(command.split(" ")).decode("utf-8").strip()


def count_cpus():
    cpus = None
    if JOB_ID:
        try:
            slurm_cpus = run_command(f"squeue --job {JOB_ID} -o %c").split("\n")[1]
            cpus = int(slurm_cpus)
        except subprocess.CalledProcessError:
            cpus = os.cpu_count()
    else:
        cpus = os.cpu_count()

    return cpus


def count_gpus():
    gpus = 0
    if JOB_ID:
        try:
            slurm_gpus = run_command(f"squeue --job {JOB_ID} -o %b").split("\n")[1]
            gpus = re.findall(r".*(\d+)", slurm_gpus) or 0
            gpus = int(gpus[0]) if gpus != 0 else gpus
        except subprocess.CalledProcessError:
            gpus = torch.cuda.device_count()
    else:
        gpus = torch.cuda.device_count()

    return gpus


def set_cpus_to_workers(config, silent=None):
    if not config.get("no_cpus_to_workers"):
        cpus = count_cpus()
        gpus = count_gpus()
        nw = config["optim"].get("num_workers")

        if cpus is not None:
            if gpus == 0:
                workers = cpus - 1
            else:
                workers = cpus // gpus

            if (silent is False or not config.get("silent")) and (nw) != workers:
                print(
                    f"🏭 Overriding num_workers from {nw}",
                    f"to {workers} to match the machine's CPUs.",
                    "Use --no_cpus_to_workers=true to disable this behavior.",
                )

            config["optim"]["num_workers"] = workers
    return config


def load_matbench_train_val_indices(fold, val_frac):
    import matbench

    fold_str = f"fold_{fold}"
    mb_val = matbench.metadata.mbv01_validation["splits"]["matbench_mp_e_form"]
    assert fold_str in mb_val

    indices = np.array(
        [int(float(i.split("-")[-1])) for i in mb_val[fold_str]["train"]]
    )

    with temp_seed(fold):
        perm = np.random.permutation(len(indices))

    n_val = int(len(indices) * val_frac)
    val_indices = indices[perm[:n_val]]
    train_indices = indices[perm[n_val:]]

    return train_indices, val_indices
