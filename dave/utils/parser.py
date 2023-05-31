from argparse import ArgumentParser
import ast


def parse_value(value):
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def dict_set_recursively(dictionary, key_sequence, val):
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        arg = arg.strip("--")
        parts = arg.split("=") if "=" in arg else (arg, "True")
        if len(parts) == 2:
            keys_concat, val = parts
        elif len(parts) > 2:
            keys_concat, val = parts[0], "=".join(parts[1:])
        else:
            raise ValueError(f"Invalid argument {arg}")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def parse_args_to_dict() -> dict:
    """
    Parse arbitrary command line arguments to a dictionary.

        Returns:
            dict: command-line args as dictionary
    """
    parser = ArgumentParser()
    # Run config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="String like {model}-{task} where {model}.yaml "
        + "must exist in configs/models and {task}.yaml must exist in configs/tasks",
    )
    # Weights and Biases
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Name for the wandb run. If not specified, it will be created by run.py",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Dave-MP20",
        help="wandb project to log to",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated tags for wandb",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="mila-ocp",
        help="wandb entity where the project is",
    )
    parser.add_argument(
        "--wandb_note",
        type=str,
        default="",
        help="wandb note for the run",
    )
    # Parse args
    args, override_args = parser.parse_known_args()
    from dave.utils.misc import merge_dicts

    return merge_dicts(dict(vars(args)), create_dict_from_args(override_args))
