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
from dataclasses import dataclass

from gt4py.next.common import Field
from model.common.tests import field_type_aliases as fa

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


@dataclass
class DiagnosticStateNonHydro:
    """Data class containing diagnostic fields that are calculated in the dynamical core (SolveNonHydro)."""

    vt: fa.EKfloatField
    vn_ie: Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: Field[
        [CellDim, KDim], float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    theta_v_ic: fa.CKfloatField
    exner_pr: fa.CKfloatField
    rho_ic: fa.CKfloatField
    ddt_exner_phy: fa.CKfloatField
    grf_tend_rho: fa.CKfloatField
    grf_tend_thv: fa.CKfloatField
    grf_tend_w: fa.CKfloatField
    mass_fl_e: fa.EKfloatField
    ddt_vn_phy: fa.EKfloatField
    grf_tend_vn: fa.EKfloatField
    ddt_vn_apc_ntl1: fa.EKfloatField
    ddt_vn_apc_ntl2: fa.EKfloatField
    ddt_w_adv_ntl1: fa.CKfloatField
    ddt_w_adv_ntl2: fa.CKfloatField

    # Analysis increments
    rho_incr: fa.EKfloatField  # moist density increment [kg/m^3]
    vn_incr: fa.EKfloatField  # normal velocity increment [m/s]
    exner_incr: fa.EKfloatField  # exner increment [- ]
    exner_dyn_incr: fa.CKfloatField  # exner pressure dynamics increment

    @property
    def ddt_vn_apc_pc(
        self,
    ) -> tuple[fa.EKfloatField, fa.EKfloatField]:
        return (self.ddt_vn_apc_ntl1, self.ddt_vn_apc_ntl2)

    @property
    def ddt_w_adv_pc(
        self,
    ) -> tuple[fa.CKfloatField, fa.CKfloatField]:
        return (self.ddt_w_adv_ntl1, self.ddt_w_adv_ntl2)


@dataclass
class InterpolationState:
    """Represents the ICON interpolation state used in the dynamical core (SolveNonhydro)."""

    e_bln_c_s: Field[[CEDim], float]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: Field[[CEDim], float]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: Field[
        [CellDim, C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: Field[[CellDim, C2E2CODim], float]
    geofac_grg_y: Field[
        [CellDim, C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: Field[[EdgeDim], float]  # Nudgeing coeffients for edges

    c_lin_e: Field[[EdgeDim, E2CDim], float]
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float]
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float]
    c_intp: Field[[VertexDim, V2CDim], float]
    geofac_rot: Field[[VertexDim, V2EDim], float]
    pos_on_tplane_e_1: Field[[ECDim], float]
    pos_on_tplane_e_2: Field[[ECDim], float]
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float]


@dataclass
class MetricStateNonHydro:
    """Dataclass containing metric fields needed in dynamical core (SolveNonhydro)."""

    bdy_halo_c: fa.CboolField
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: Field[[CellDim, KDim], bool]
    rayleigh_w: Field[[KDim], float]

    wgtfac_c: fa.CKfloatField
    wgtfacq_c: fa.CKfloatField
    wgtfac_e: fa.EKfloatField
    wgtfacq_e: fa.EKfloatField

    exner_exfac: fa.CKfloatField
    exner_ref_mc: fa.CKfloatField
    rho_ref_mc: fa.CKfloatField
    theta_ref_mc: fa.CKfloatField
    rho_ref_me: fa.EKfloatField
    theta_ref_me: fa.EKfloatField
    theta_ref_ic: fa.CKfloatField

    d_exner_dz_ref_ic: fa.CKfloatField
    ddqz_z_half: fa.CKfloatField  # half KDim ?
    d2dexdz2_fac1_mc: fa.CKfloatField
    d2dexdz2_fac2_mc: fa.CKfloatField
    ddxn_z_full: fa.EKfloatField
    ddqz_z_full_e: fa.EKfloatField
    ddxt_z_full: fa.EKfloatField
    inv_ddqz_z_full: fa.CKfloatField

    vertoffset_gradp: Field[[ECDim, KDim], float]
    zdiff_gradp: Field[[ECDim, KDim], float]
    ipeidx_dsl: fa.EKboolField
    pg_exdist: fa.EKfloatField

    vwind_expl_wgt: fa.CfloatField
    vwind_impl_wgt: fa.CfloatField

    hmask_dd3d: Field[[EdgeDim], float]
    scalfac_dd3d: Field[[KDim], float]

    coeff1_dwdz: fa.CKfloatField
    coeff2_dwdz: fa.CKfloatField
    coeff_gradekin: Field[[ECDim], float]


@dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: fa.EKfloatField
    mass_flx_me: fa.EKfloatField
    mass_flx_ic: fa.CKfloatField
    vol_flx_ic: fa.CKfloatField
