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

from icon4py.model.common import field_type_aliases as fa
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

    vt: fa.EdgeKField[float]
    vn_ie: Field[
        [EdgeDim, KDim], float
    ]  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s] # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    w_concorr_c: Field[
        [CellDim, KDim], float
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
    ddt_vn_apc_ntl1: fa.EdgeKField[float]
    ddt_vn_apc_ntl2: fa.EdgeKField[float]
    ddt_w_adv_ntl1: fa.CellKField[float]
    ddt_w_adv_ntl2: fa.CellKField[float]

    # Analysis increments
    rho_incr: fa.EdgeKField[float]  # moist density increment [kg/m^3]
    vn_incr: fa.EdgeKField[float]  # normal velocity increment [m/s]
    exner_incr: fa.EdgeKField[float]  # exner increment [- ]
    exner_dyn_incr: fa.CellKField[float]  # exner pressure dynamics increment

    @property
    def ddt_vn_apc_pc(
        self,
    ) -> tuple[fa.EdgeKField[float], fa.EdgeKField[float]]:
        return (self.ddt_vn_apc_ntl1, self.ddt_vn_apc_ntl2)

    @property
    def ddt_w_adv_pc(
        self,
    ) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
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
    nudgecoeff_e: fa.EdgeField[float]  # Nudgeing coeffients for edges

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
    ddqz_z_half: fa.CellKField[float]  # half KDim ?
    d2dexdz2_fac1_mc: fa.CellKField[float]
    d2dexdz2_fac2_mc: fa.CellKField[float]
    ddxn_z_full: fa.EdgeKField[float]
    ddqz_z_full_e: fa.EdgeKField[float]
    ddxt_z_full: fa.EdgeKField[float]
    inv_ddqz_z_full: fa.CellKField[float]

    vertoffset_gradp: Field[[ECDim, KDim], float]
    zdiff_gradp: Field[[ECDim, KDim], float]
    ipeidx_dsl: fa.EdgeKField[bool]
    pg_exdist: fa.EdgeKField[float]

    vwind_expl_wgt: fa.CellField[float]
    vwind_impl_wgt: fa.CellField[float]

    hmask_dd3d: fa.EdgeField[float]
    scalfac_dd3d: fa.KField[float]

    coeff1_dwdz: fa.CellKField[float]
    coeff2_dwdz: fa.CellKField[float]
    coeff_gradekin: Field[[ECDim], float]


@dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: fa.EdgeKField[float]
    mass_flx_me: fa.EdgeKField[float]
    mass_flx_ic: fa.CellKField[float]
    vol_flx_ic: fa.CellKField[float]
