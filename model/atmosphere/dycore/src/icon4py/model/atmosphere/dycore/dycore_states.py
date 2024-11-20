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

    vt: fa.EdgeKField[float]
    vn_ie: fa.EdgeKField[
        float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: fa.CellKField[
        float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    theta_v_ic: fa.CellKField[float]
    exner_pr: fa.CellKField[float]
    rho_ic: fa.CellKField[float]
    ddt_exner_phy: fa.CellKField[float]
    grf_tend_rho: fa.CellKField[float]
    grf_tend_thv: fa.CellKField[float]
    grf_tend_w: fa.CellKField[float]
    mass_fl_e: fa.EdgeKField[float]
    ddt_vn_phy: fa.EdgeKField[float]
    grf_tend_vn: fa.EdgeKField[float]
    ddt_vn_apc_pc: common_utils.Pair[fa.EdgeKField[float]]
    ddt_w_adv_pc: common_utils.Pair[fa.CellKField[float]]

    # Analysis increments
    rho_incr: Optional[fa.EdgeKField[float]]  # moist density increment [kg/m^3]
    vn_incr: Optional[fa.EdgeKField[float]]  # normal velocity increment [m/s]
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
    rho_ref_me: fa.EdgeKField[float]
    theta_ref_me: fa.EdgeKField[float]
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
    ipeidx_dsl: fa.EdgeKField[bool]
    pg_exdist: fa.EdgeKField[float]

    vwind_expl_wgt: fa.CellField[float]
    vwind_impl_wgt: fa.CellField[float]

    hmask_dd3d: fa.EdgeField[float]
    scalfac_dd3d: fa.KField[float]

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
