import numpy as np
from gt4py.next import Dimension, NeighborTableOffsetProvider


def neighbortable_offset_provider_for_1d_sparse_fields(
    old_shape: tuple[int, int],
    origin_axis: Dimension,
    neighbor_axis: Dimension,
):
    table = np.arange(old_shape[0] * old_shape[1]).reshape(old_shape)
    return NeighborTableOffsetProvider(table, origin_axis, neighbor_axis, table.shape[1])
