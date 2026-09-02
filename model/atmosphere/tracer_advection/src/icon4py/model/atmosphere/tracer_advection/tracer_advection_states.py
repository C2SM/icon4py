# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass(frozen=True)
class AdvectionDiagnosticState:
    """Represents the diagnostic fields needed in tracer_advection."""

    #: mass of air in layer at physics time step now [kg/m^2]
    airmass_now: fa.CellKField[ta.wpfloat]

    #: mass of air in layer at physics time step new [kg/m^2]
    airmass_new: fa.CellKField[ta.wpfloat]

    #: tracer tendency field for use in grid refinement [kg/kg/s]
    grf_tend_tracer: fa.CellKField[ta.wpfloat]

    #: horizontal tracer flux at edges [kg/m/s]
    hfl_tracer: fa.EdgeKField[ta.wpfloat]

    #: vertical tracer flux at cells [kg/m/s]
    vfl_tracer: fa.CellKHalfField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionPrepAdvState:
    """Represents the prepare tracer_advection state needed in tracer_advection."""

    #: horizontal velocity at edges for computation of backward trajectories averaged over dynamics substeps [m/s]
    vn_traj: fa.EdgeKField[ta.wpfloat]

    #: mass flux at full level edges averaged over dynamics substeps [kg/m^2/s]
    mass_flx_me: fa.EdgeKField[ta.wpfloat]

    #: mass flux at half level centers averaged over dynamics substeps [kg/m^2/s]
    mass_flx_ic: fa.CellKHalfField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionInterpolationState:
    """Represents the interpolation state needed in tracer_advection."""

    #: factor for divergence
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat]

    #: coefficients used for rbf interpolation of the tangential velocity component
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat]

    #: x-components of positions of various points on local plane tangential to the edge midpoint
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]

    #: y-components of positions of various points on local plane tangential to the edge midpoint
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionLeastSquaresState:
    """Represents the least squares state needed in tracer_advection."""

    #: pseudo (or Moore-Penrose) inverse of lsq design matrix A
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionMetricState:
    """Represents the metric fields needed in tracer_advection.

    The deep-atmosphere modification factors below are all 1 in the shallow atmosphere,
    which is the only mode icon4py supports (the dycore rejects 'deepatmos_mode', see
    'solve_nonhydro.NonHydrostaticConfig'). ICON does the same: it initialises them to 1
    in 'mo_nonhydro_state.f90' and only overwrites them inside the 'IF (ldeepatmo)'
    branch of 'mo_vertical_grid.f90'. They are kept as fields, rather than folded away,
    because the ICON stencils they feed take them unconditionally.
    """

    #: metrical modification factor for horizontal part of divergence at full levels (dims.KDim)
    deepatmo_divh: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (dims.KDim)
    deepatmo_divzl: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (dims.KDim)
    deepatmo_divzu: fa.KField[ta.wpfloat]

    #: vertical grid spacing at full levels
    ddqz_z_full: fa.CellKField[ta.wpfloat]


def initialize_advection_diagnostic_state(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
) -> AdvectionDiagnosticState:
    return AdvectionDiagnosticState(
        airmass_now=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        airmass_new=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        grf_tend_tracer=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        hfl_tracer=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        vfl_tracer=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KHalfDim,
            allocator=allocator,
            dtype=ta.wpfloat,
        ),
    )
