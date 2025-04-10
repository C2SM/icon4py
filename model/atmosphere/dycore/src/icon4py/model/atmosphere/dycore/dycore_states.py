# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Optional

import gt4py.next as gtx

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    utils as common_utils,
)


@dataclasses.dataclass
class DiagnosticStateNonHydro:
    """Data class containing diagnostic fields that are calculated in the dynamical core (SolveNonHydro)."""

    tangential_wind: fa.EdgeKField[float]
    """
    Declared as vt in ICON. Tangential wind at edge.
    """

    vn_on_half_levels: fa.EdgeKField[
        float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as vn_ie in ICON. Normal wind at edge on k-half levels.
    """

    contravariant_correction_at_cells_on_half_levels: fa.CellKField[
        float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as w_concorr_c in ICON. Contravariant correction at cell center on k-half levels. vn dz/dn + vt dz/dt, z is topography height
    """

    theta_v_at_cells_on_half_levels: fa.CellKField[float]
    """
    Declared as theta_v_ic in ICON.
    """

    exner_pr: fa.CellKField[float]
    rho_ic: fa.CellKField[float]
    ddt_exner_phy: fa.CellKField[float]
    grf_tend_rho: fa.CellKField[float]
    grf_tend_thv: fa.CellKField[float]
    grf_tend_w: fa.CellKField[float]
    mass_fl_e: fa.EdgeKField[float]
    normal_wind_tendency_due_to_physics_process: fa.EdgeKField[float]
    """
    Declared as ddt_vn_phy in ICON.
    """
    grf_tend_vn: fa.EdgeKField[float]
    normal_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.EdgeKField[float]]
    """
    Declared as ddt_vn_apc_pc in ICON. Advective tendency of normal wind (including coriolis force).
    """

    vertical_wind_advective_tendency: common_utils.PredictorCorrectorPair[fa.CellKField[float]]
    """
    Declared as ddt_w_adv_pc in ICON. Advective tendency of vertical wind.
    """

    # Analysis increments
    rho_incr: Optional[fa.EdgeKField[float]]  # moist density increment [kg/m^3]
    normal_wind_iau_increments: Optional[fa.EdgeKField[float]]  # normal velocity increment [m/s]
    """
    Declared as vn_incr in ICON.
    """
    exner_incr: Optional[fa.EdgeKField[float]]  # exner increment [- ]
    exner_dyn_incr: fa.CellKField[float]  # exner pressure dynamics increment


@dataclasses.dataclass
class InterpolationState:
    """Represents the ICON interpolation state used in the dynamical core (SolveNonhydro)."""

    e_bln_c_s: gtx.Field[
        gtx.Dims[dims.CEDim], float
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[
        gtx.Dims[dims.CEDim], float
    ]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], float]
    geofac_grg_y: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[float]  # Nudgeing coeffients for edges

    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float]
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], float]
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], float]
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], float]
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], float]
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.ECDim], float]
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.ECDim], float]
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], float]


@dataclasses.dataclass
class MetricStateNonHydro:
    """Dataclass containing metric fields needed in dynamical core (SolveNonhydro)."""

    bdy_halo_c: fa.CellField[bool]
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: fa.CellKField[bool]
    rayleigh_w: fa.KField[float]

    wgtfac_c: fa.CellKField[float]
    wgtfacq_c: fa.CellKField[float]
    wgtfac_e: fa.EdgeKField[float]
    wgtfacq_e: fa.EdgeKField[float]

    exner_exfac: fa.CellKField[float]
    exner_ref_mc: fa.CellKField[float]
    rho_ref_mc: fa.CellKField[float]
    theta_ref_mc: fa.CellKField[float]
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[float]
    """
    Declared as rho_ref_me in ICON.
    """
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[float]
    """
    Declared as theta_ref_me in ICON.
    """
    theta_ref_ic: fa.CellKField[float]

    d_exner_dz_ref_ic: fa.CellKField[float]
    ddqz_z_half: fa.CellKField[float]  # half dims.KDim ?
    d2dexdz2_fac1_mc: fa.CellKField[float]
    d2dexdz2_fac2_mc: fa.CellKField[float]
    ddxn_z_full: fa.EdgeKField[float]
    ddqz_z_full_e: fa.EdgeKField[float]
    ddxt_z_full: fa.EdgeKField[float]
    inv_ddqz_z_full: fa.CellKField[float]

    vertoffset_gradp: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], float]
    zdiff_gradp: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], float]
    pg_edgeidx_dsl: fa.EdgeKField[bool]
    pg_exdist: fa.EdgeKField[float]

    vwind_expl_wgt: fa.CellField[float]
    vwind_impl_wgt: fa.CellField[float]

    horizontal_mask_for_3d_divdamp: fa.EdgeField[float]
    """
    Declared as hmask_dd3d in ICON. A horizontal mask where 3D divergence is computed for the divergence damping.
    3D divergence is defined as divergence of horizontal wind plus vertical derivative of vertical wind (dw/dz).
    """
    scaling_factor_for_3d_divdamp: fa.KField[float]
    """
    Declared as scalfac_dd3d in ICON. A scaling factor in vertical dimension for 3D divergence damping. 
    """

    coeff1_dwdz: fa.CellKField[float]
    coeff2_dwdz: fa.CellKField[float]
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], float]


@dataclasses.dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: fa.EdgeKField[float]
    mass_flx_me: fa.EdgeKField[float]
    mass_flx_ic: fa.CellKField[float]
    vol_flx_ic: fa.CellKField[float]
