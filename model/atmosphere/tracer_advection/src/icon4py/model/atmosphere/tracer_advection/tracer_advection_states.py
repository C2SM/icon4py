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

from icon4py.model.atmosphere.tracer_advection import weno_least_squares
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
    vfl_tracer: fa.CellKField[ta.wpfloat]  # TODO(dastrm): should be KHalfDim


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
class AdvectionQuadraticState:
    """Represents the quadratic (miura3) reconstruction state (ihadv_tracer=3).

    The same fields as 'AdvectionWenoQuadraticState' except that the pseudoinverse is the
    single full-stencil one rather than 27 candidates, so there is no smoothness weighting
    and hence no cell area.
    """

    # pseudoinverse coefficients on the direct neighbor rows, [5]
    lsq_pseudoinv_direct: tuple[gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat], ...]

    # pseudoinverse coefficients on the butterfly rows, [5]
    lsq_pseudoinv_butterfly: tuple[
        gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat], ...
    ]

    # cell averages of the monomials [x, y, x^2, y^2, xy]
    lsq_moments_1: fa.CellField[ta.wpfloat]
    lsq_moments_2: fa.CellField[ta.wpfloat]
    lsq_moments_3: fa.CellField[ta.wpfloat]
    lsq_moments_4: fa.CellField[ta.wpfloat]
    lsq_moments_5: fa.CellField[ta.wpfloat]

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

    # the 27 linear weights l_weights_s the type-VI candidates were assembled with
    # (weno_least_squares.linear_weights); the run-time blend must use the same vector,
    # which is why they travel with the pseudoinverses. Default: the live optimised set.
    l_weights_s: tuple[ta.wpfloat, ...] = tuple(
        ta.wpfloat(w) for w in weno_least_squares.L_WEIGHTS_S
    )


@dataclasses.dataclass(frozen=True)
class AdvectionWenoHybridState:
    """Represents the hybrid quadratic / quadratic-WENO state (ihadv_tracer=132).

    The hybrid reconstructs every cell with the full 9-point pseudoinverse of miura3 and
    decides from the residual of that fit whether to keep it or to blend the 27 WENO
    candidates, so it carries both states plus ICON's 'lsq_error' (the transposed weighted
    design matrix, REAL(sp) in the Fortran, i.e. ta.fortran_sp_float here;
    weno_least_squares.compute_lsq_error_quadratic)
    scattered onto the C2E2C / C2E2C2E2C rows like the pseudoinverses, and the mask of the
    butterfly slots that carry a stencil cell (compute_butterfly_slot_mask).

    Both states must come from one least-squares setup (the same 9-point stencil, moments
    and geometry, see driver_utils._construct_weno_hybrid_state): the full-stencil
    pseudoinverse of 'quadratic_state' is the matrix the type-VI candidates 1-3 of
    'weno_quadratic_state' are assembled from (with its l_weights_s), and 'lsq_error' is
    the design matrix of that same fit. '__post_init__' checks the part of this that is
    cheap to check: the moment and geometry fields of the two states are the same objects.
    That check is by identity, not by value: two states built separately from the same
    data (e.g. both read from savepoints) are rejected even if their fields are equal.
    """

    weno_quadratic_state: AdvectionWenoQuadraticState
    quadratic_state: AdvectionQuadraticState

    # lsq_error rows on the direct neighbours, [5 unknowns]; REAL(sp) in the Fortran, see
    # type_alias.fortran_sp_float
    lsq_error_direct: tuple[
        gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.fortran_sp_float],
        ...,
    ]

    # lsq_error rows on the butterfly slots, [5 unknowns]
    lsq_error_butterfly: tuple[
        gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.fortran_sp_float],
        ...,
    ]

    # 1 on the butterfly slots holding an outer stencil cell, 0 on the padding slots
    # (weno_least_squares.compute_butterfly_slot_mask as an int32 field)
    lsq_butterfly_active: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.int32]

    def __post_init__(self) -> None:
        shared = (
            "lsq_moments_1",
            "lsq_moments_2",
            "lsq_moments_3",
            "lsq_moments_4",
            "lsq_moments_5",
            "pos_on_tplane_e_1_x",
            "pos_on_tplane_e_2_x",
            "pos_on_tplane_e_1_y",
            "pos_on_tplane_e_2_y",
            "edge_verts_1_x",
            "edge_verts_2_x",
            "edge_verts_1_y",
            "edge_verts_2_y",
            "tangent_orientation",
        )
        not_shared = [
            name
            for name in shared
            if getattr(self.quadratic_state, name) is not getattr(self.weno_quadratic_state, name)
        ]
        if not_shared:
            raise ValueError(
                "'AdvectionWenoHybridState' needs the quadratic and the quadratic-WENO state "
                "built from one least-squares setup, but these fields are not the same "
                f"objects: {', '.join(not_shared)}."
            )


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
        # vertical flux at cell half levels: one more level than KDim
        vfl_tracer=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            extend={dims.KDim: 1},
            allocator=allocator,
            dtype=ta.wpfloat,
        ),
    )
