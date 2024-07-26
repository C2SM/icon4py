# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import dataclasses

import gt4py.next as gtx

from icon4py.model.common.dimension import (
    C2E2CODim,
    CEDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)


@dataclasses.dataclass
class DiagnosticStateNonHydro:
    """Data class containing diagnostic fields that are calculated in the dynamical core (SolveNonHydro)."""

    vt: gtx.Field[[EdgeDim, KDim], float]
    vn_ie: gtx.Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: gtx.Field[
        [CellDim, KDim], float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    theta_v_ic: gtx.Field[[CellDim, KDim], float]
    exner_pr: gtx.Field[[CellDim, KDim], float]
    rho_ic: gtx.Field[[CellDim, KDim], float]
    ddt_exner_phy: gtx.Field[[CellDim, KDim], float]
    grf_tend_rho: gtx.Field[[CellDim, KDim], float]
    grf_tend_thv: gtx.Field[[CellDim, KDim], float]
    grf_tend_w: gtx.Field[[CellDim, KDim], float]
    mass_fl_e: gtx.Field[[EdgeDim, KDim], float]
    ddt_vn_phy: gtx.Field[[EdgeDim, KDim], float]
    grf_tend_vn: gtx.Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl1: gtx.Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl2: gtx.Field[[EdgeDim, KDim], float]
    ddt_w_adv_ntl1: gtx.Field[[CellDim, KDim], float]
    ddt_w_adv_ntl2: gtx.Field[[CellDim, KDim], float]

    # Analysis increments
    rho_incr: gtx.Field[[EdgeDim, KDim], float]  # moist density increment [kg/m^3]
    vn_incr: gtx.Field[[EdgeDim, KDim], float]  # normal velocity increment [m/s]
    exner_incr: gtx.Field[[EdgeDim, KDim], float]  # exner increment [- ]
    exner_dyn_incr: gtx.Field[[CellDim, KDim], float]  # exner pressure dynamics increment

    @property
    def ddt_vn_apc_pc(
        self,
    ) -> tuple[gtx.Field[[EdgeDim, KDim], float], gtx.Field[[EdgeDim, KDim], float]]:
        return (self.ddt_vn_apc_ntl1, self.ddt_vn_apc_ntl2)

    @property
    def ddt_w_adv_pc(
        self,
    ) -> tuple[gtx.Field[[CellDim, KDim], float], gtx.Field[[CellDim, KDim], float]]:
        return (self.ddt_w_adv_ntl1, self.ddt_w_adv_ntl2)


@dataclasses.dataclass
class InterpolationState:
    """Represents the ICON interpolation state used in the dynamical core (SolveNonhydro)."""

    e_bln_c_s: gtx.Field[
        [CEDim], float
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[[CEDim], float]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        [CellDim, C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[[CellDim, C2E2CODim], float]
    geofac_grg_y: gtx.Field[
        [CellDim, C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: gtx.Field[[EdgeDim], float]  # Nudgeing coeffients for edges

    c_lin_e: gtx.Field[[EdgeDim, E2CDim], float]
    geofac_grdiv: gtx.Field[[EdgeDim, E2C2EODim], float]
    rbf_vec_coeff_e: gtx.Field[[EdgeDim, E2C2EDim], float]
    c_intp: gtx.Field[[VertexDim, V2CDim], float]
    geofac_rot: gtx.Field[[VertexDim, V2EDim], float]
    pos_on_tplane_e_1: gtx.Field[[ECDim], float]
    pos_on_tplane_e_2: gtx.Field[[ECDim], float]
    e_flx_avg: gtx.Field[[EdgeDim, E2C2EODim], float]


@dataclasses.dataclass
class MetricStateNonHydro:
    """Dataclass containing metric fields needed in dynamical core (SolveNonhydro)."""

    bdy_halo_c: gtx.Field[[CellDim], bool]
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: gtx.Field[[CellDim, KDim], bool]
    rayleigh_w: gtx.Field[[KDim], float]

    wgtfac_c: gtx.Field[[CellDim, KDim], float]
    wgtfacq_c: gtx.Field[[CellDim, KDim], float]
    wgtfac_e: gtx.Field[[EdgeDim, KDim], float]
    wgtfacq_e: gtx.Field[[EdgeDim, KDim], float]

    exner_exfac: gtx.Field[[CellDim, KDim], float]
    exner_ref_mc: gtx.Field[[CellDim, KDim], float]
    rho_ref_mc: gtx.Field[[CellDim, KDim], float]
    theta_ref_mc: gtx.Field[[CellDim, KDim], float]
    rho_ref_me: gtx.Field[[EdgeDim, KDim], float]
    theta_ref_me: gtx.Field[[EdgeDim, KDim], float]
    theta_ref_ic: gtx.Field[[CellDim, KDim], float]

    d_exner_dz_ref_ic: gtx.Field[[CellDim, KDim], float]
    ddqz_z_half: gtx.Field[[CellDim, KDim], float]  # half KDim ?
    d2dexdz2_fac1_mc: gtx.Field[[CellDim, KDim], float]
    d2dexdz2_fac2_mc: gtx.Field[[CellDim, KDim], float]
    ddxn_z_full: gtx.Field[[EdgeDim, KDim], float]
    ddqz_z_full_e: gtx.Field[[EdgeDim, KDim], float]
    ddxt_z_full: gtx.Field[[EdgeDim, KDim], float]
    inv_ddqz_z_full: gtx.Field[[CellDim, KDim], float]

    vertoffset_gradp: gtx.Field[[ECDim, KDim], float]
    zdiff_gradp: gtx.Field[[ECDim, KDim], float]
    ipeidx_dsl: gtx.Field[[EdgeDim, KDim], bool]
    pg_exdist: gtx.Field[[EdgeDim, KDim], float]

    vwind_expl_wgt: gtx.Field[[CellDim], float]
    vwind_impl_wgt: gtx.Field[[CellDim], float]

    hmask_dd3d: gtx.Field[[EdgeDim], float]
    scalfac_dd3d: gtx.Field[[KDim], float]

    coeff1_dwdz: gtx.Field[[CellDim, KDim], float]
    coeff2_dwdz: gtx.Field[[CellDim, KDim], float]
    coeff_gradekin: gtx.Field[[ECDim], float]


@dataclasses.dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: gtx.Field[[EdgeDim, KDim], float]
    mass_flx_me: gtx.Field[[EdgeDim, KDim], float]
    mass_flx_ic: gtx.Field[[CellDim, KDim], float]
    vol_flx_ic: gtx.Field[[CellDim, KDim], float]
