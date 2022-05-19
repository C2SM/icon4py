import numpy as np
from functional.iterator.embedded import np_as_located_field


def random_field(mesh, *dims):
    return np_as_located_field(*dims)(
        np.random.randn(*map(lambda x: mesh.size[x], dims))
    )


def zero_field(mesh, *dims):
    return np_as_located_field(*dims)(
        np.zeros(shape=tuple(map(lambda x: mesh.size[x], dims)))
    )


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
