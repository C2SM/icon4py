# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""State constructors of the tmx surface integration datatests, built from the
``tmx-surface-*`` savepoints of the exp.exclaim_ape_aesPhys archive.

The reference archive is the aqua-planet (ocean-only) run, so only the ocean
tile is active; the ice/land inputs of ``SurfaceInputState`` are zero-filled and
the ice ``SurfaceState`` fields stay at their allocation zeros."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface import surface_states
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


def construct_surface_input_state(
    entry_savepoint: sb.TmxSurfaceEntrySavepoint,
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator | None,
) -> surface_states.SurfaceInputState:
    """Build a ``SurfaceInputState`` from the ``tmx-surface-entry`` savepoint.

    Only the ocean-tile fraction and surface temperature are present in the
    aqua-planet archive; the ice/land tile fractions, the prescribed land fluxes
    and the ice radiative forcing are absent and zero-filled (they do not enter
    the ocean exchange path)."""

    def zero() -> data_alloc.NDArray:
        return data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator)

    present = {
        "ta": entry_savepoint.ta(),
        "qa": entry_savepoint.qa(),
        "ua": entry_savepoint.ua(),
        "va": entry_savepoint.va(),
        "pa": entry_savepoint.pa(),
        "psfc": entry_savepoint.psfc(),
        "rho_atm": entry_savepoint.rho_atm(),
        "dz": entry_savepoint.dz(),
        "emissivity": entry_savepoint.emissivity(),
        "snowfall": entry_savepoint.snowfall(),
        "ice_thickness": entry_savepoint.ice_thickness(),
        "ocean_u": entry_savepoint.ocean_u(),
        "ocean_v": entry_savepoint.ocean_v(),
        "ice_u": entry_savepoint.ice_u(),
        "ice_v": entry_savepoint.ice_v(),
        "sst": entry_savepoint.tsfc_oce(),
        "frac_oce": entry_savepoint.frac_oce(),
    }
    fields = {
        f.name: present.get(f.name, zero())
        for f in dataclasses.fields(surface_states.SurfaceInputState)
    }
    return surface_states.SurfaceInputState(**fields)
