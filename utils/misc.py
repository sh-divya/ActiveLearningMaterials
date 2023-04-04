from os.path import expandvars
from pathlib import Path


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

