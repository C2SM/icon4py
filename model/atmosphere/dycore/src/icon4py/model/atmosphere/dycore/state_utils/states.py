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

from icon4py.model.common.dimension import (
    CellDim,
    EdgeDim,
    KDim,
    CEDim,
    VertexDim,
    V2EDim,
    C2E2CODim,
    E2CDim,
    E2C2EODim,
    E2C2EDim,
    V2CDim,
    ECDim,
)


@dataclass
class DiagnosticStateNonHydro:
    vt: Field[[EdgeDim, KDim], float]
    vn_ie: Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: Field[
        [CellDim, KDim], float
    ]  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    theta_v_ic: Field[[CellDim, KDim], float]
    exner_pr: Field[[CellDim, KDim], float]
    rho_ic: Field[[CellDim, KDim], float]
    ddt_exner_phy: Field[[CellDim, KDim], float]
    grf_tend_rho: Field[[CellDim, KDim], float]
    grf_tend_thv: Field[[CellDim, KDim], float]
    grf_tend_w: Field[[CellDim, KDim], float]
    mass_fl_e: Field[[EdgeDim, KDim], float]
    ddt_vn_phy: Field[[EdgeDim, KDim], float]
    grf_tend_vn: Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float]
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float]
    ddt_w_adv_ntl1: Field[[CellDim, KDim], float]
    ddt_w_adv_ntl2: Field[[CellDim, KDim], float]

    # Analysis increments
    rho_incr: Field[[EdgeDim, KDim], float]  # moist density increment [kg/m^3]
    vn_incr: Field[[EdgeDim, KDim], float]  # normal velocity increment [m/s]
    exner_incr: Field[[EdgeDim, KDim], float]  # exner increment [- ]

    @property
    def ddt_vn_apc_pc(
        self,
    ) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
        return (self.ddt_vn_apc_ntl1, self.ddt_vn_apc_ntl2)

    @property
    def ddt_w_adv_pc(
        self,
    ) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
        return (self.ddt_w_adv_ntl1, self.ddt_w_adv_ntl2)


@dataclass
class InterpolationState:
    """Represents the ICON interpolation state used int SolveNonHydro."""

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
    bdy_halo_c: Field[[CellDim], bool]
    # Finally, a mask field that excludes boundary halo points
    mask_prog_halo_c: Field[[CellDim, KDim], bool]
    rayleigh_w: Field[[KDim], float]

    wgtfac_c: Field[[CellDim, KDim], float]
    wgtfacq_c: Field[[CellDim, KDim], float]
    wgtfac_e: Field[[EdgeDim, KDim], float]
    wgtfacq_e_dsl: Field[[EdgeDim, KDim], float]

    exner_exfac: Field[[CellDim, KDim], float]
    exner_ref_mc: Field[[CellDim, KDim], float]
    rho_ref_mc: Field[[CellDim, KDim], float]
    theta_ref_mc: Field[[CellDim, KDim], float]
    rho_ref_me: Field[[EdgeDim, KDim], float]
    theta_ref_me: Field[[EdgeDim, KDim], float]
    theta_ref_ic: Field[[CellDim, KDim], float]

    d_exner_dz_ref_ic: Field[[CellDim, KDim], float]
    ddqz_z_half: Field[[CellDim, KDim], float]  # half KDim ?
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float]
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float]
    ddxn_z_full: Field[[EdgeDim, KDim], float]
    ddqz_z_full_e: Field[[EdgeDim, KDim], float]
    ddxt_z_full: Field[[EdgeDim, KDim], float]
    inv_ddqz_z_full: Field[[CellDim, KDim], float]

    vertoffset_gradp: Field[[ECDim, KDim], float]
    zdiff_gradp: Field[[ECDim, KDim], float]
    ipeidx_dsl: Field[[EdgeDim, KDim], bool]
    pg_exdist: Field[[EdgeDim, KDim], float]

    vwind_expl_wgt: Field[[CellDim], float]
    vwind_impl_wgt: Field[[CellDim], float]

    hmask_dd3d: Field[[EdgeDim], float]
    scalfac_dd3d: Field[[KDim], float]

    coeff1_dwdz: Field[[CellDim, KDim], float]
    coeff2_dwdz: Field[[CellDim, KDim], float]
    coeff_gradekin: Field[[ECDim], float]
