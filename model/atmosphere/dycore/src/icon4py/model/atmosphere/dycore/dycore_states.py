# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum

import gt4py.next as gtx
from gt4py.eve.utils import FrozenNamespace

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
    utils as common_utils,
)


class TimeSteppingScheme(enum.IntEnum):
    """Parameter called `itime_scheme` in ICON namelist."""

    #: Contravariant vertical velocity is computed in the predictor step only, velocity tendencies are computed in the corrector step only
    MOST_EFFICIENT = 4
    #: Contravariant vertical velocity is computed in both substeps (beneficial for numerical stability in very-high resolution setups with extremely steep slopes)
    STABLE = 5
    #:  As STABLE, but velocity tendencies are also computed in both substeps (no benefit, but more expensive)
    EXPENSIVE = 6


class DivergenceDampingType(enum.IntEnum):
    #: divergence damping acting on 2D divergence
    TWO_DIMENSIONAL = 2
    #: divergence damping acting on 3D divergence
    THREE_DIMENSIONAL = 3
    #: combination of 3D div.damping in the troposphere with transition to 2D div. damping in the stratosphere
    COMBINED = 32


class DivergenceDampingOrder(FrozenNamespace[int]):
    #: 2nd order divergence damping
    SECOND_ORDER = 2
    #: 4th order divergence damping
    FOURTH_ORDER = 4
    #: combined 2nd and 4th orders divergence damping and enhanced vertical wind off - centering during initial spinup phase
    COMBINED = 24


class HorizontalPressureDiscretizationType(FrozenNamespace[int]):
    """Parameter called igradp_method in ICON namelist."""

    #: conventional discretization with metric correction term
    CONVENTIONAL = 1
    #: Taylor-expansion-based reconstruction of pressure
    TAYLOR = 2
    #: Similar discretization as igradp_method_taylor, but uses hydrostatic approximation for downward extrapolation over steep slopes
    TAYLOR_HYDRO = 3
    #: Cubic / quadratic polynomial interpolation for pressure reconstruction
    POLYNOMIAL = 4
    #: Same as igradp_method_polynomial, but hydrostatic approximation for downward extrapolation over steep slopes
    POLYNOMIAL_HYDRO = 5


class RhoThetaAdvectionType(FrozenNamespace[int]):
    """Parameter called iadv_rhotheta in ICON namelist."""

    #: simple 2nd order upwind-biased scheme
    SIMPLE = 1
    #: 2nd order Miura horizontal
    MIURA = 2


@dataclasses.dataclass
class DiagnosticStateNonHydro:
    """Data class containing diagnostic fields that are calculated in the dynamical core (SolveNonHydro)."""

    max_vertical_cfl: ta.wpfloat
    """
    Declared as max_vcfl_dyn in ICON. Maximum vertical CFL number over all substeps.
    """

    tangential_wind: fa.EdgeKField[ta.vpfloat]
    """
    Declared as vt in ICON. Tangential wind at edge.
    """

    vn_on_half_levels: fa.EdgeKField[
        ta.vpfloat
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as vn_ie in ICON. Normal wind at edge on k-half levels.
    """

    contravariant_correction_at_cells_on_half_levels: fa.CellKField[
        ta.vpfloat
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as w_concorr_c in ICON. Contravariant correction at cell center on k-half levels. vn dz/dn + vt dz/dt, z is topography height
    """

    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat]
    """
    Declared as theta_v_ic in ICON.
    """

    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat]
    """
    Declared as exner_pr in ICON.
    """

    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat]
    """
    Declared as rho_ic in ICON.
    """

    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat]
    """
    Declared as ddt_exner_phy in ICON.
    """
    grf_tend_rho: fa.CellKField[ta.wpfloat]
    grf_tend_thv: fa.CellKField[ta.wpfloat]
    grf_tend_w: fa.CellKField[ta.wpfloat]
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat]
    """
    Declared as mass_fl_e in ICON.
    """
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat]
    """
    Declared as ddt_vn_phy in ICON.
    """

    grf_tend_vn: fa.EdgeKField[ta.wpfloat]
    normal_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.EdgeKField[ta.vpfloat]]
    """
    Declared as ddt_vn_apc_pc in ICON. Advective tendency of normal wind (including coriolis force).
    """

    vertical_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.CellKField[ta.vpfloat]]
    """
    Declared as ddt_w_adv_pc in ICON. Advective tendency of vertical wind.
    """

    # Analysis increments
    rho_iau_increment: fa.CellKField[ta.vpfloat]  # moist density increment [kg/m^3]
    """
    Declared as rho_incr in ICON.
    """
    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat]  # normal velocity increment [m/s]
    """
    Declared as vn_incr in ICON.
    """
    exner_iau_increment: fa.CellKField[ta.vpfloat]  # exner increment [- ]
    """
    Declared as exner_incr in ICON.
    """
    exner_dynamical_increment: fa.CellKField[ta.vpfloat]  # exner pressure dynamics increment
    """
    Declared as exner_dyn_incr in ICON.
    """


@dataclasses.dataclass
class InterpolationState:
    """Represents the ICON interpolation state used in the dynamical core (SolveNonhydro)."""

    e_bln_c_s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat
    ]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat]
    geofac_grg_y: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[ta.wpfloat]  # Nudgeing coeffients for edges

    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat]
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat]
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat]
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat]
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat]


@dataclasses.dataclass
class MetricStateNonHydro:
    """Dataclass containing metric fields needed in dynamical core (SolveNonhydro)."""

    bdy_halo_c: fa.CellField[bool]
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: fa.CellKField[bool]
    rayleigh_w: fa.KField[ta.wpfloat]

    wgtfac_c: fa.CellKField[ta.vpfloat]
    wgtfacq_c: fa.CellKField[ta.vpfloat]
    wgtfac_e: fa.EdgeKField[ta.vpfloat]
    wgtfacq_e: fa.EdgeKField[ta.vpfloat]

    time_extrapolation_parameter_for_exner: fa.CellKField[ta.vpfloat]
    """
    Declared as exner_exfac in ICON.
    """
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as exner_ref_mc in ICON.
    """
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as rho_ref_mc in ICON.
    """
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as theta_ref_mc in ICON.
    """
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat]
    """
    Declared as rho_ref_me in ICON.
    """
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat]
    """
    Declared as theta_ref_me in ICON.
    """
    reference_theta_at_cells_on_half_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as theta_ref_ic in ICON.
    """
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as d_exner_dz_ref_ic in ICON.
    """
    ddqz_z_half: fa.CellKField[ta.vpfloat]  # dims.KHalfDim
    d2dexdz2_fac1_mc: fa.CellKField[ta.vpfloat]
    d2dexdz2_fac2_mc: fa.CellKField[ta.vpfloat]
    ddxn_z_full: fa.EdgeKField[ta.vpfloat]
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat]
    ddxt_z_full: fa.EdgeKField[ta.vpfloat]
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat]

    vertoffset_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32]
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], ta.vpfloat]
    pg_edgeidx_dsl: fa.EdgeKField[bool]
    pg_exdist: fa.EdgeKField[ta.vpfloat]

    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat]
    """
    Declared as vwind_expl_wgt in ICON. The explicitness parameter for exner and w in the vertically
    implicit dycore solver.
    exner_w_explicit_weight_parameter = 1 - exner_w_implicit_weight_parameter
    """
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat]
    """
    Declared as vwind_impl_wgt in ICON. The implicitness parameter for exner and w in the vertically
    implicit dycore solver. It is denoted as eta below eq. 3.20 in ICON tutorial 2023. However,
    it is only vwind_offctr that can be set via namelist. The actual computation of
    exner_w_implicit_weight_parameter is not shown in the tutorial.
    """

    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat]
    """
    Declared as hmask_dd3d in ICON. A horizontal mask where 3D divergence is computed for the divergence damping.
    3D divergence is defined as divergence of horizontal wind plus vertical derivative of vertical wind (dw/dz).
    """
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat]
    """
    Declared as scalfac_dd3d in ICON. A scaling factor in vertical dimension for 3D divergence damping.
    """

    coeff1_dwdz: fa.CellKField[ta.vpfloat]
    coeff2_dwdz: fa.CellKField[ta.vpfloat]
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat]


@dataclasses.dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: fa.EdgeKField[ta.wpfloat]
    mass_flx_me: fa.EdgeKField[ta.wpfloat]
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat]
    """
    Declared as mass_flx_ic in ICON.
    """
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat]
    """
    Declared as vol_flx_ic in ICON.
    """
