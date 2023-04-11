import copy
import os
from itertools import product
from os.path import expandvars
from pathlib import Path
from typing import Dict, List, Union
from uuid import uuid4

from yaml import dump

JOB_ID = os.environ.get("SLURM_JOB_ID")


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
                ], default=0
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
    config.pop("xscale", None)
    config.pop("yscale", None)
    print(dump(config))


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


def merge_dicts(dict1: dict, dict2: dict) -> dict:
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

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            elif isinstance(v, list) and isinstance(dict1[k], list):
                if len(dict1[k]) != len(dict2[k]):
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
                        merge_dicts(d1, d2) for d1, d2 in zip(dict1[k], v)
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
