import numpy as np
import spglib
from pyxtal.symmetry import Group
from scipy.spatial.transform import Rotation as R


def get_symmetries(space_group: int) -> dict:
    """
    Get {"rotations": np.array(3 dims), "translations": np.array(2 dims)}
    from space group number (int)
    """
    assert 1 <= space_group <= 230
    return spglib.get_symmetry_from_database(
        Group(space_group, style="spglib").hall_number
    )


def symmetry_vector(rot: np.array, trans: np.array) -> np.array:
    """
    Convert rotation matrix and translation vector to symmetry vector
    defined as [q1, q2, q3, q4, t1, t2, t3]

    Args:
        rot (np.array): Rotation matrix
        trans (np.array): Translation vector

    Returns:
        np.array: Symmetry vector [q1, q2, q3, q4, t1, t2, t3]
    """
    quat = R.from_matrix(rot).as_quat()
    return np.concatenate([quat, trans])


def all_symmetry_vectors(space_group: int) -> np.array:
    """
    Get all symmetry vectors from space group number.

    Args:
        space_group (int): Space group number

    Returns:
        np.array: All symmetry vectors as [N, 7] 2D arrray
    """
    symmetries = get_symmetries(space_group)
    symmetries = [
        (symmetries["rotations"][k], symmetries["translations"][k])
        for k in range(len(symmetries["rotations"]))
    ]
    return np.concatenate(
        [symmetry_vector(rot, trans)[None, :] for rot, trans in symmetries], axis=0
    )
