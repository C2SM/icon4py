# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.common.grid import gridfile
from icon4py.model.common.utils import data_allocation as data_alloc


def valid_e2c_neighbors(
    e2c: data_alloc.NDArray,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute per-slot validity masks and a safe-to-index version of the E2C connectivity.

    Boundary edges of a limited-area grid have only one cell neighbor. Depending on
    how the connectivity was constructed, the missing second neighbor is either
    ``INVALID_INDEX`` or a duplicate of the first one (an interior edge always has
    two distinct neighbors), so both spellings must be treated as invalid.

    Args:
        e2c: edge to cell connectivity, shape (num_edges, 2)

    Returns:
        valid_neighbor_0: bool mask, True where the first neighbor exists
        valid_neighbor_1: bool mask, True where the second neighbor exists
        safe_e2c: e2c with invalid entries replaced by 0, safe for indexing
    """
    array_ns = data_alloc.array_namespace(e2c)
    missing = gridfile.GridFile.INVALID_INDEX
    valid_neighbor_0 = e2c[:, 0] != missing
    valid_neighbor_1 = (e2c[:, 1] != missing) & (e2c[:, 1] != e2c[:, 0])
    safe_e2c = array_ns.where(e2c != missing, e2c, 0)
    return valid_neighbor_0, valid_neighbor_1, safe_e2c


def revert_repeated_index_to_invalid(offset: np.ndarray):
    num_elements = offset.shape[0]
    for i in range(num_elements):
        # convert repeated indices back into -1
        for val in np.flip(offset[i, :]):
            if np.count_nonzero(val == offset[i, :]) > 1:
                unique_values, counts = np.unique(offset[i, :], return_counts=True)
                rep_values = unique_values[counts > 1]
                rep_indices = np.where(np.isin(offset[i, :], rep_values))[0]
                offset[i, rep_indices[1:]] = gridfile.GridFile.INVALID_INDEX
    return offset
