# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Soil energy setup: vertical geometry and thermal properties.

Host-side (one-time) construction of the fields the soil temperature kernels
consume, matching the JSBACH SSE `jsbach_lite` + tmx configuration
(l_heat_cap_dyn = l_heat_cond_dyn = l_heat_cap_map = l_heat_cond_map = F, i.e. the
static FAO path). The dynamic, moisture-dependent property path
(calc_vol_heat_capacity / calc_thermal_conductivity) is not used by that config and
is not ported here.

Per the port's gather policy, per-cell index gathers (the FAO table lookup) live in
this host layer, not in the stencils.
"""

import dataclasses

import gt4py.next as gtx
import numpy as np
import numpy.typing as npt

from icon4py.model.common import dimension as dims


# FAO soil-type thermal tables, index 0..6 (6 = invalid), mo_sse_init.f90:44-45.
FAO_VOLUMETRIC_HEAT_CAPACITY = (2.25e6, 1.93e6, 2.10e6, 2.25e6, 2.36e6, 2.48e6, -1.0)
FAO_THERMAL_DIFFUSIVITY = (7.4e-7, 8.7e-7, 8.0e-7, 7.4e-7, 7.1e-7, 6.7e-7, -1.0)


@dataclasses.dataclass(frozen=True)
class SoilThermalGrid:
    """Vertical geometry of the `soil_depth_energy` grid (mo_sse_config_class.f90:238-256).

    dz, mids, bots are per layer; zd1 is the inverse spacing between adjacent layer
    mid-depths, with the (unused) bottom entry set to zero.
    """

    dz: gtx.Field  # layer thickness [m]
    mids: gtx.Field  # layer mid-depths [m]
    bots: gtx.Field  # layer bottom depths [m]
    zd1: gtx.Field  # 1 / (mids(k+1) - mids(k)), bottom entry 0 [1/m]


def soil_thermal_grid(layer_bottom_depths: npt.NDArray) -> SoilThermalGrid:
    """Build the soil energy vertical geometry from the layer bottom depths (`soillev`).

    Args:
        layer_bottom_depths: depth of each layer's lower boundary [m] (the `soillev`
            array read from ic_land_soil.nc), length nsoil.

    Returns:
        the per-layer thickness, mid-depths, bottom depths and inverse mid-spacing.
    """
    depths = np.empty(len(layer_bottom_depths) + 1, dtype=np.float64)
    depths[0] = 0.0
    depths[1:] = layer_bottom_depths
    dz = depths[1:] - depths[:-1]
    mids = 0.5 * (depths[:-1] + depths[1:])
    bots = depths[1:].copy()
    zd1 = np.zeros_like(mids)
    zd1[:-1] = 1.0 / (mids[1:] - mids[:-1])
    return SoilThermalGrid(
        dz=gtx.as_field((dims.KDim,), dz),
        mids=gtx.as_field((dims.KDim,), mids),
        bots=gtx.as_field((dims.KDim,), bots),
        zd1=gtx.as_field((dims.KDim,), zd1),
    )


def fao_soil_thermal_properties(
    fao_index: npt.NDArray, num_levels: int
) -> tuple[gtx.Field, gtx.Field]:
    """Per-cell soil volumetric heat capacity and conductivity from the FAO index.

    Static FAO path (mo_sse_init.f90 sse_init_bc): a per-cell lookup into the FAO
    tables followed by heat_cond = vol_heat_cap * thermal_diffusivity; with
    l_heat_cap_dyn = l_heat_cond_dyn = F the per-cell values are used unchanged for
    every soil layer, so they are broadcast here.

    Args:
        fao_index: per-cell FAO soil-type index (rounded to nearest, matching NINT).
        num_levels: number of soil layers to broadcast across.

    Returns:
        (vol_heat_cap [J/m^3/K], heat_cond [W/m/K]) as CellK fields.
    """
    index = np.rint(fao_index).astype(np.int64)
    vol_heat_cap = np.asarray(FAO_VOLUMETRIC_HEAT_CAPACITY)[index]
    heat_cond = vol_heat_cap * np.asarray(FAO_THERMAL_DIFFUSIVITY)[index]
    vol_heat_cap_sl = np.ascontiguousarray(
        np.broadcast_to(vol_heat_cap[:, np.newaxis], (vol_heat_cap.shape[0], num_levels))
    )
    heat_cond_sl = np.ascontiguousarray(
        np.broadcast_to(heat_cond[:, np.newaxis], (heat_cond.shape[0], num_levels))
    )
    return (
        gtx.as_field((dims.CellDim, dims.KDim), vol_heat_cap_sl),
        gtx.as_field((dims.CellDim, dims.KDim), heat_cond_sl),
    )
