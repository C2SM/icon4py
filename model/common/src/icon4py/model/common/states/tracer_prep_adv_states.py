# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
TracerPrepAdvState for tracer advection.

It is allocated by the driver, filled by the initial condition and mutated by the
tracer advection, so it lives here and not in the tracer advection package.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@dataclasses.dataclass(frozen=True)
class TracerPrepAdvState:
    """Represents the prepare tracer_advection state needed in tracer_advection."""

    #: horizontal velocity at edges for computation of backward trajectories averaged over dynamics substeps [m/s]
    vn_traj: fa.EdgeKField[ta.wpfloat]

    #: mass flux at full level edges averaged over dynamics substeps [kg/m^2/s]
    mass_flx_me: fa.EdgeKField[ta.wpfloat]

    #: mass flux at half level centers averaged over dynamics substeps [kg/m^2/s]
    mass_flx_ic: fa.CellKHalfField[ta.wpfloat]


def initialize_tracer_prep_adv_state(
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator,
) -> TracerPrepAdvState:
    vn_traj = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    mass_flx_me = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    mass_flx_ic = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator, dtype=ta.wpfloat
    )
    return TracerPrepAdvState(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
    )
