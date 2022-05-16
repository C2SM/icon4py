import numpy as np


def add_kdim(arr: np.array, k: int) -> np.array:
    """Adds a k dimension to a numpy array.

    Args:
        arr: numpy array.
        k: number of k levels.

    Returns:
        Array with added k dimension.
    """
    return np.repeat(arr[None, :], k, axis=0).T


def get_cell_to_k_table(k_arr, k):
    """Creates cell to k table based on an input array and k value.

    Args:
        k_arr: 1D input array holding k values.
        k: k value to use within k table.

    Returns:
        2D array filled with k values
    """
    # creating cell to k table
    c2k = np.expand_dims(k_arr, axis=-1)
    return np.repeat(c2k[:], k, axis=-1)
