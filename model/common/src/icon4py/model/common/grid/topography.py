# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


def create(experiment_name: str, cell_lat: data_alloc.NDArray) -> data_alloc.NDArray:
    match experiment_name:
        case "exclaim_nh35_tri_jws":
            return _exclaim_nh35_tri_jws(cell_lat)
        case "exclaim_gauss3d":
            raise NotImplementedError("Gauss3d topography is not yet implemented")
        case _:
            raise ValueError(f"Unknown experiment name for topography: {experiment_name!r}")


def _exclaim_nh35_tri_jws(cell_lat: data_alloc.NDArray) -> data_alloc.NDArray:
    u0 = 35.0
    array_ns = data_alloc.array_namespace(cell_lat)
    sin_lat = array_ns.sin(cell_lat)
    cos_lat = array_ns.cos(cell_lat)

    eta_0 = 0.252

    fac1 = u0 * array_ns.cos((1.0 - eta_0) * (math.pi / 2)) ** 1.5
    fac2 = (-2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0) + 1.0 / 6.3) * fac1
    fac3 = (
        (1.6 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.5 * (math.pi / 2))
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    topo_c = fac1 * (fac2 + fac3) / phy_const.GRAV

    return topo_c


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
    *,
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    exchange: decomposition.ExchangeRuntime,
    num_iterations: int = 25,
) -> data_alloc.NDArray:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """
    array_ns = data_alloc.array_namespace(topography)
    smooth_topo = topography.copy()
    # TODO(@halungge): if the input topopgraphy is properly exchanged, which it should this is not needed here.
    exchange.exchange(dims.CellDim, smooth_topo, stream=decomposition.BLOCK)

    for _ in range(num_iterations):
        nabla2_topo = compute_nabla2_on_cell(smooth_topo, geofac_n2s, c2e2co)
        array_ns.add(smooth_topo, 0.125 * nabla2_topo * cell_areas, out=smooth_topo)

        exchange.exchange(dims.CellDim, smooth_topo, stream=decomposition.BLOCK)

    return smooth_topo
