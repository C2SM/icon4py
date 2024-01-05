import numpy as np

from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.utils import scale_k, set_zero_v_k


def test_scale_k():
    grid = SimpleGrid()
    field = random_field(grid, KDim)
    scaled_field = zero_field(grid, KDim)
    factor = 2.0
    scale_k(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * field.asnumpy(), scaled_field.asnumpy())


def test_set_zero_vertex_k(backend):
    grid = SimpleGrid()
    f = random_field(grid, VertexDim, KDim)
    set_zero_v_k(f, offset_provider={})
    assert np.allclose(0.0, f.asnumpy())
