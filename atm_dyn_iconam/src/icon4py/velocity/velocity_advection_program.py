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


from functional.common import Field
from functional.ffront.decorator import program
from functional.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_01 import (
    _mo_velocity_advection_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_02 import (
    _mo_velocity_advection_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_03 import (
    _mo_velocity_advection_stencil_03,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_04 import (
    _mo_velocity_advection_stencil_04,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_05 import (
    _mo_velocity_advection_stencil_05,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_06 import (
    _mo_velocity_advection_stencil_06,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_07 import (
    _mo_velocity_advection_stencil_07,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_08 import (
    _mo_velocity_advection_stencil_08,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_09 import (
    _mo_velocity_advection_stencil_09,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_10 import (
    _mo_velocity_advection_stencil_10,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_11 import (
    _mo_velocity_advection_stencil_11,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_12 import (
    _mo_velocity_advection_stencil_12,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_13 import (
    _mo_velocity_advection_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_14 import (
    _mo_velocity_advection_stencil_14,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_15 import (
    _mo_velocity_advection_stencil_15,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_16 import (
    _mo_velocity_advection_stencil_16,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_17 import (
    _mo_velocity_advection_stencil_17,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_18 import (
    _mo_velocity_advection_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_19 import (
    _mo_velocity_advection_stencil_19,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_20 import (
    _mo_velocity_advection_stencil_20,
)
from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.state_utils.utils import set_zero_w_k


@program(backend=gtfn_cpu.run_gtfn)
def velocity_advection_run(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    vt: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    w: Field[[CellDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    local_z_w_v: Field[[VertexDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_ekinh: Field[[CellDim, KDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    local_cfl_clipping: Field[[CellDim, KDim], float],
    local_pre_levelmask: Field[[CellDim, KDim], float],
    local_vcfl: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
    z_w_con_c_full: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    levelmask: Field[[KDim], bool],
    owner_mask: Field[[CellDim], bool],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    scalfac_exdiff: float,
    coeff_gradekin: Field[[ECDim], float],
    zeta: Field[[VertexDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    ddt_vn_adv: Field[[EdgeDim, KDim], float],
    area_edge: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    cell_startindex_nudging: int,
    cell_endindex_local_minus1: int,
    cell_endindex_local: int,
    edge_startindex_nudging_plus_one: int,
    edge_endindex_local: int,
    edge_endindex_local_minus2: int,
    edge_endindex_local_minus1: int,
    nflatlev_startindex: int,
    nrdmax_startindex: int,
    nlev: int,
    nlevp1: int,
):

    _mo_velocity_advection_stencil_01(
        vn,
        rbf_vec_coeff_e,
        out=vt,
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_02(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_kin_hor_e),
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (1, nlev),
        },
    )
    _mo_velocity_advection_stencil_03(
        wgtfac_e,
        vt,
        out=z_vt_ie,
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (1, nlev),
        },
    )
    _mo_velocity_advection_stencil_04(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (nflatlev_startindex, nlev),
        },
    )
    _mo_velocity_advection_stencil_05(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (0, 0),
        },
    )
    _mo_velocity_advection_stencil_06(
        wgtfacq_e,
        vn,
        out=vn_ie,
        domain={
            EdgeDim: (5, edge_endindex_local_minus2),
            KDim: (nlev, nlev),
        },
    )
    _mo_velocity_advection_stencil_07(
        vn_ie,
        inv_dual_edge_length,
        w,
        z_vt_ie,
        inv_primal_edge_length,
        tangent_orientation,
        local_z_w_v,
        out=local_z_v_grad_w,
        domain={
            EdgeDim: (5, edge_endindex_local_minus1),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_08(
        z_kin_hor_e,
        e_bln_c_s,
        out=local_z_ekinh,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_09(
        z_w_concorr_me,
        e_bln_c_s,
        out=local_z_w_concorr_mc,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex, nlev),
        },
    )
    _mo_velocity_advection_stencil_10(
        local_z_w_concorr_mc,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex + 1, nlev),
        },
    )
    _mo_velocity_advection_stencil_11(
        w,
        out=local_z_w_con_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_12(
        out=local_z_w_con_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nlevp1, nlevp1),
        },
    )
    _mo_velocity_advection_stencil_13(
        local_z_w_con_c,
        w_concorr_c,
        out=local_z_w_con_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex + 1, nlev),
        },
    )

    set_zero_w_k(local_cfl_clipping, offset_provider={})
    set_zero_w_k(local_pre_levelmask, offset_provider={})
    set_zero_w_k(local_vcfl, offset_provider={})

    _mo_velocity_advection_stencil_14(
        ddqz_z_half,
        local_z_w_con_c,
        local_cfl_clipping,
        local_pre_levelmask,
        local_vcfl,
        cfl_w_limit,
        dtime,
        out=(local_cfl_clipping, local_pre_levelmask, local_vcfl, local_z_w_con_c),
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (max(3, nrdmax_startindex - 2), nlev - 3),
        },
    )
    _mo_velocity_advection_stencil_15(
        local_z_w_con_c,
        out=z_w_con_c_full,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_16(
        local_z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        out=ddt_w_adv,
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (1, nlev),
        },
    )
    _mo_velocity_advection_stencil_17(
        e_bln_c_s,
        local_z_v_grad_w,
        ddt_w_adv,
        out=ddt_w_adv,
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (1, nlev),
        },
    )
    _mo_velocity_advection_stencil_18(
        levelmask,
        local_cfl_clipping,
        owner_mask,
        local_z_w_con_c,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        ddt_w_adv,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        out=ddt_w_adv,
        domain={},
    )
    _mo_velocity_advection_stencil_19(
        z_kin_hor_e,
        coeff_gradekin,
        local_z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        out=ddt_vn_adv,
        domain={
            EdgeDim: (edge_startindex_nudging_plus_one, edge_endindex_local),
            KDim: (0, nlev),
        },
    )
    _mo_velocity_advection_stencil_20(
        levelmask,
        c_lin_e,
        z_w_con_c_full,
        ddqz_z_full_e,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        zeta,
        geofac_grdiv,
        vn,
        ddt_vn_adv,
        cfl_w_limit,
        scalfac_exdiff,
        dtime,
        out=ddt_vn_adv[:, :-1],
        domain={
            EdgeDim: (edge_startindex_nudging_plus_one, edge_endindex_local),
            KDim: (max(3, nrdmax_startindex - 2), nlev - 4),
        },
    )
