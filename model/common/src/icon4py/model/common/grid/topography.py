# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils.compute_nabla2_on_cell import compute_nabla2_on_cell
from icon4py.model.common.utils import data_allocation as data_alloc

import numpy as np


@gtx.field_operator
def _update_smoothed_topography(
    smoothed_topography: fa.CellField[ta.wpfloat],
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:
    """
    Updates the smoothed topography field inside the loop.
    """
    return smoothed_topography + 0.125 * nabla2_topo * cell_areas


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_smoothed_topography(
    nabla2_topo: fa.CellField[ta.wpfloat],
    cell_areas: fa.CellField[ta.wpfloat],
    smoothed_topography: fa.CellField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _update_smoothed_topography(
        smoothed_topography=smoothed_topography,
        nabla2_topo=nabla2_topo,
        cell_areas=cell_areas,
        out=smoothed_topography,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )

def compute_nabla2_on_cell_numpy(
    psi_c: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co:data_alloc.NDArray,
) -> np.ndarray:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres. (Numpy version)
    """
    nabla2_psi_c =np.sum(psi_c[c2e2co] * geofac_n2s, axis=1)
    return nabla2_psi_c

def update_smoothed_topography_numpy(
    smoothed_topography: np.ndarray,
    nabla2_topo: np.ndarray,
    cell_areas: np.ndarray,
) -> np.ndarray:
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
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    smoothed_topography = topography.copy()

    for _ in range(num_iterations):

        nabla2_topo=compute_nabla2_on_cell_numpy(smoothed_topography, geofac_n2s, c2e2co)
        smoothed_topography = update_smoothed_topography_numpy(smoothed_topography, nabla2_topo, cell_areas)

    return smoothed_topography

