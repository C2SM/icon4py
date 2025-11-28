# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from types import ModuleType

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_nabla2_on_cell(
    psi_c: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres.
    """
    nabla2_psi_c = array_ns.sum(psi_c[c2e2co] * geofac_n2s, axis=1)
    return nabla2_psi_c


def smooth_topography(
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    exchange: Callable[[gtx.Dimension, gtx.Field], None],
    num_iterations: int = 25,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """
    # as field _will_ do a copy. The call to ndarray.copy here is to make it explicit that we need a copy.
    topo_as_field = gtx.as_field((dims.CellDim,), topography.copy(), dtype=topography.dtype)
    # TODO(@halungge): if the input topopgraphy is properly exchanged, which it should this is not needed here.
    exchange(topo_as_field.domain.dims[0], topo_as_field)

    for _ in range(num_iterations):
        nabla2_topo = compute_nabla2_on_cell(topo_as_field.ndarray, geofac_n2s, c2e2co, array_ns)
        array_ns.add(
            topo_as_field.ndarray, 0.125 * nabla2_topo * cell_areas, out=topo_as_field.ndarray
        )

        exchange(topo_as_field.domain.dims[0], topo_as_field)

    return topo_as_field.ndarray
