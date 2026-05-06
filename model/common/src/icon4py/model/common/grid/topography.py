# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_nabla2_on_cell(
    psi_c: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres.
    """
    array_ns = data_alloc.array_namespace(psi_c)
    nabla2_psi_c = array_ns.sum(psi_c[c2e2co] * geofac_n2s, axis=1)
    return nabla2_psi_c


def smooth_topography(
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    exchange: Callable[[data_alloc.NDArray], None],
    num_iterations: int = 25,
) -> data_alloc.NDArray:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """
    array_ns = data_alloc.array_namespace(topography)
    smooth_topo = topography.copy()
    # TODO(@halungge): if the input topopgraphy is properly exchanged, which it should this is not needed here.
    exchange(smooth_topo)

    for _ in range(num_iterations):
        nabla2_topo = compute_nabla2_on_cell(smooth_topo, geofac_n2s, c2e2co)
        array_ns.add(smooth_topo, 0.125 * nabla2_topo * cell_areas, out=smooth_topo)

        exchange(smooth_topo)

    return smooth_topo
