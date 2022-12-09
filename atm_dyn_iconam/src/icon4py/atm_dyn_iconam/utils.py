import numpy as np
from functional.common import Dimension
from functional.iterator.embedded import np_as_located_field


def zero_field(mesh, *dims: Dimension, dtype=float):
    shapex =  tuple(map(lambda x: mesh.size[x], dims))
    return np_as_located_field(*dims)(np.zeros(shapex, dtype=dtype))
