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
        keys_concat, val = arg.split("=") if "=" in arg else (arg, "True")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def parse_args_to_dict() -> dict:
    """
    Parse arbitrary command line arguments to a dictionary.

        Returns:
            dict: _description_
    """
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    return create_dict_from_args(override_args)
