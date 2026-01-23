# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math
from types import ModuleType

import numpy as np

from icon4py.model.common.grid import gridfile
from icon4py.model.common.utils import data_allocation as data_alloc


def revert_repeated_index_to_invalid(offset: np.ndarray, array_ns: ModuleType):
    num_elements = offset.shape[0]
    for i in range(num_elements):
        # convert repeated indices back into -1
        for val in array_ns.flip(offset[i, :]):
            if array_ns.count_nonzero(val == offset[i, :]) > 1:
                unique_values, counts = array_ns.unique(offset[i, :], return_counts=True)
                rep_values = unique_values[counts > 1]
                rep_indices = array_ns.where(array_ns.isin(offset[i, :], rep_values))[0]
                offset[i, rep_indices[1:]] = gridfile.GridFile.INVALID_INDEX
    return offset


def compute_sqrt(
    input_val: data_alloc.NDArray,
) -> float:
    """
    compute the sqrt value of input_val.
    """
    sqrt_val = math.sqrt(input_val)
    return sqrt_val
