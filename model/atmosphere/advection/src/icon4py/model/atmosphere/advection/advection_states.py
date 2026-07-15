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
    """Represents the diagnostic fields needed in advection."""

    #: mass of air in layer at physics time step now [kg/m^2]
    airmass_now: fa.CellKField[ta.wpfloat]

    #: mass of air in layer at physics time step new [kg/m^2]
    airmass_new: fa.CellKField[ta.wpfloat]

    #: tracer tendency field for use in grid refinement [kg/kg/s]
    grf_tend_tracer: fa.CellKField[ta.wpfloat]

    #: horizontal tracer flux at edges [kg/m/s]
    hfl_tracer: fa.EdgeKField[ta.wpfloat]

    #: vertical tracer flux at cells [kg/m/s]
    vfl_tracer: fa.CellKField[ta.wpfloat]  # TODO(dastrm): should be KHalfDim


@dataclasses.dataclass(frozen=True)
class AdvectionPrepAdvState:
    """Represents the prepare advection state needed in advection."""

    #: horizontal velocity at edges for computation of backward trajectories averaged over dynamics substeps [m/s]
    vn_traj: fa.EdgeKField[ta.wpfloat]

    #: mass flux at full level edges averaged over dynamics substeps [kg/m^2/s]
    mass_flx_me: fa.EdgeKField[ta.wpfloat]

    #: mass flux at half level centers averaged over dynamics substeps [kg/m^2/s]
    mass_flx_ic: fa.CellKField[ta.wpfloat]  # TODO(dastrm): should be KHalfDim


@dataclasses.dataclass(frozen=True)
class AdvectionInterpolationState:
    """Represents the interpolation state needed in advection."""

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
    """Represents the least squares state needed in advection."""

    #: pseudo (or Moore-Penrose) inverse of lsq design matrix A
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionWenoLinearState:
    """Represents the linear WENO least squares state (ihadv_tracer=102).

    The zonal/meridional candidate pseudoinverses over the C2E2C rows, one per
    linear WENO candidate.
    """

    lsq_pseudoinv_zonal_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_zonal_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_zonal_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_meridional_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_meridional_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_meridional_c3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionWenoQuadraticState:
    """Represents the quadratic (miura3) WENO state (ihadv_tracer=103).

    The 27 candidate pseudoinverses (unknowns [x, y, x^2, y^2, xy]) are split
    over the C2E2C and C2E2C2E2C rows by weno_least_squares.scatter_to_offsets
    and stored as nested tuples indexed [candidate][unknown], so the runtime
    candidate loop can slice per candidate. Also carries the torus ffsl
    backtrajectory geometry ('compute_ffsl_backtrajectory' inputs beyond
    standard grid state) and the quadrature/smoothness cell fields.
    """

    # candidate pseudoinverse coefficients on the direct neighbor rows, [27][5]
    lsq_pseudoinv_direct: tuple[
        tuple[gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat], ...], ...
    ]

    # candidate pseudoinverse coefficients on the butterfly rows, [27][5]
    lsq_pseudoinv_butterfly: tuple[
        tuple[gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat], ...], ...
    ]

    # cell averages of the monomials [x, y, x^2, y^2, xy]
    lsq_moments_1: fa.CellField[ta.wpfloat]
    lsq_moments_2: fa.CellField[ta.wpfloat]
    lsq_moments_3: fa.CellField[ta.wpfloat]
    lsq_moments_4: fa.CellField[ta.wpfloat]
    lsq_moments_5: fa.CellField[ta.wpfloat]

    # cell area, used by the smoothness indicator
    cell_area: fa.CellField[ta.wpfloat]

    # E2C cell centers in the edge-local frame (pos_on_tplane_e components 1:2)
    pos_on_tplane_e_1_x: fa.EdgeField[ta.wpfloat]
    pos_on_tplane_e_2_x: fa.EdgeField[ta.wpfloat]
    pos_on_tplane_e_1_y: fa.EdgeField[ta.wpfloat]
    pos_on_tplane_e_2_y: fa.EdgeField[ta.wpfloat]

    # E2V vertices in the edge-local frame (pos_on_tplane_e components 3:4)
    edge_verts_1_x: fa.EdgeField[ta.wpfloat]
    edge_verts_2_x: fa.EdgeField[ta.wpfloat]
    edge_verts_1_y: fa.EdgeField[ta.wpfloat]
    edge_verts_2_y: fa.EdgeField[ta.wpfloat]

    # primal/dual normal components on the E2C cells (the per-edge normals on the torus)
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    dual_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    dual_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]

    # edge orientation, needed for the counterclockwise indicator lvn_sys_pos
    tangent_orientation: fa.EdgeField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionMetricState:
    """Represents the metric fields needed in advection."""

    #: metrical modification factor for horizontal part of divergence at full levels (KDim)
    deepatmo_divh: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
    deepatmo_divzl: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
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
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
    )
