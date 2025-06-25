# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

import numpy as np

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


def update_smoothed_topography(
    smoothed_topography: np.ndarray,
    nabla2_topo: np.ndarray,
    cell_areas: np.ndarray,
) -> data_alloc.NDArray:
    """
    Updates the smoothed topography field inside the loop. (Numpy version)
    """
    return smoothed_topography + 0.125 * nabla2_topo * cell_areas


def smooth_topography(
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    num_iterations: int = 25,
) -> data_alloc.NDArray:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    smoothed_topography = topography.copy()

    for _ in range(num_iterations):
        nabla2_topo = compute_nabla2_on_cell(smoothed_topography, geofac_n2s, c2e2co)
        smoothed_topography = update_smoothed_topography(
            smoothed_topography, nabla2_topo, cell_areas
        )

    return smoothed_topography
