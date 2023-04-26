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
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import maximum, where
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_01 import (
    _mo_velocity_advection_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_02 import (
    _mo_velocity_advection_stencil_02,
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
from icon4py.state_utils.utils import _set_bool_c_k, _set_zero_c_k


@field_operator
def _fused_stencils_1_2(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    vt = _mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e)

    (vn_ie, z_kin_hor_e) = _mo_velocity_advection_stencil_02(wgtfac_e, vn, vt)
    return vt, vn_ie, z_kin_hor_e


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_1_2(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    vt: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
):
    _fused_stencils_1_2(vn, rbf_vec_coeff_e, wgtfac_e, out=(vt, vn_ie, z_kin_hor_e))


@field_operator
def _fused_stencils_4_5_6(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    k_field: Field[[KDim], int],
    nflatlev_startindex: int,
    nlev: int,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:

    z_w_concorr_me = where(
        k_field >= nflatlev_startindex & k_field < nlev,
        _mo_velocity_advection_stencil_04(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )

    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k_field == 0,
        _mo_velocity_advection_stencil_05(vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )

    vn_ie = where(
        k_field == nlev, _mo_velocity_advection_stencil_06(wgtfacq_e, vn), vn_ie
    )

    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_4_5_6(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    k_field: Field[[KDim], int],
    edge_startindex_interior_minus2: int,
    nflatlev_startindex: int,
    nlev: int,
):
    _fused_stencils_4_5_6(
        vn,
        vt,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        k_field,
        nflatlev_startindex,
        nlev,
        out=(z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e),
        domain={EdgeDim: (4, edge_startindex_interior_minus2)},
    )


@field_operator
def _fused_stencils_9_10(
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int],
    nflatlev_startindex: int,
    nlev: int,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    local_z_w_concorr_mc = where(
        k_field >= nflatlev_startindex & k_field < nlev,
        _mo_velocity_advection_stencil_09(z_w_concorr_me, e_bln_c_s),
        local_z_w_concorr_mc,
    )

    w_concorr_c = where(
        k_field >= nflatlev_startindex + 1 & k_field < nlev,
        _mo_velocity_advection_stencil_10(local_z_w_concorr_mc, wgtfac_c),
        w_concorr_c,
    )

    return local_z_w_concorr_mc, w_concorr_c


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_9_10(
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int],
    cell_startindex_interior_minus1: int,
    nflatlev_startindex: int,
    nlev: int,
):
    _fused_stencils_9_10(
        z_w_concorr_me,
        e_bln_c_s,
        local_z_w_concorr_mc,
        wgtfac_c,
        w_concorr_c,
        k_field,
        nflatlev_startindex,
        nlev,
        out=(local_z_w_concorr_mc, w_concorr_c),
        domain={CellDim: (4, cell_startindex_interior_minus1)},
    )


@field_operator
def _fused_stencils_11_to_14(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], float],
    cfl_w_limit: float,
    dtime: float,
    nrdmax_startindex: int,
    nflatlev_startindex: int,
    nlev: int,
    nlevp1: int,
):
    local_z_w_con_c = where(
        k_field >= 0 & k_field < nlev,
        _mo_velocity_advection_stencil_11(w),
        local_z_w_con_c,
    )

    local_z_w_con_c = where(
        k_field == nlevp1, _mo_velocity_advection_stencil_12(), local_z_w_con_c
    )

    local_z_w_con_c = where(
        k_field >= (nflatlev_startindex + 1) & k_field < nlev,
        _mo_velocity_advection_stencil_13(local_z_w_con_c, w_concorr_c),
        local_z_w_con_c,
    )

    local_cfl_clipping = _set_bool_c_k()
    local_pre_levelmask = _set_bool_c_k()
    local_vcfl = _set_zero_c_k()

    k_lower = maximum(3, nrdmax_startindex - 2)
    (local_cfl_clipping, local_pre_levelmask, local_vcfl, local_z_w_con_c) = where(
        k_field >= k_lower & k_field < (nlev - 3),
        _mo_velocity_advection_stencil_14(
            ddqz_z_half,
            local_z_w_con_c,
            local_cfl_clipping,
            local_pre_levelmask,
            local_vcfl,
            cfl_w_limit,
            dtime,
        ),
        (local_cfl_clipping, local_pre_levelmask, local_vcfl, local_z_w_con_c),
    )

    return local_cfl_clipping, local_pre_levelmask, local_vcfl, local_z_w_con_c


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_11_to_14(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    local_cfl_clipping: Field[[CellDim, KDim], bool],
    local_pre_levelmask: Field[[CellDim, KDim], bool],
    local_vcfl: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], float],
    cfl_w_limit: float,
    dtime: float,
    nrdmax_startindex: int,
    cell_endindex_interior_minus1: int,
    nflatlev_startindex: int,
    nlev: int,
    nlevp1: int,
):
    _fused_stencils_11_to_14(
        w,
        w_concorr_c,
        local_z_w_con_c,
        ddqz_z_half,
        k_field,
        cfl_w_limit,
        dtime,
        nrdmax_startindex,
        nflatlev_startindex,
        nlev,
        nlevp1,
        out=(local_cfl_clipping, local_pre_levelmask, local_vcfl, local_z_w_con_c),
        domain={CellDim: (4, cell_endindex_interior_minus1)},
    )


@field_operator
def _fused_stencils_16_to_17(
    w: Field[[CellDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:

    ddt_w_adv = _mo_velocity_advection_stencil_16(
        local_z_w_con_c, w, coeff1_dwdz, coeff2_dwdz
    )

    ddt_w_adv = _mo_velocity_advection_stencil_17(
        e_bln_c_s, local_z_v_grad_w, ddt_w_adv
    )
    return ddt_w_adv


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_16_to_17(
    w: Field[[CellDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
):
    _fused_stencils_16_to_17(
        w,
        local_z_v_grad_w,
        e_bln_c_s,
        local_z_w_con_c,
        coeff1_dwdz,
        coeff2_dwdz,
        out=ddt_w_adv,
    )


@field_operator
def _fused_stencils_19_to_20(
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    local_z_ekinh: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
    z_w_con_c_full: Field[[CellDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    zeta: Field[[VertexDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    ddt_vn_adv: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    levelmask: Field[[KDim], bool],
    k_field: Field[[KDim], int],
    scalfac_exdiff: float,
    area_edge: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    nrdmax_startindex: int,
    nlev: int,
) -> Field[[EdgeDim, KDim], float]:

    ddt_vn_adv = where(
        k_field >= 0 & k_field < nlev,
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
        ),
        ddt_vn_adv,
    )
    k_lower = maximum(3, nrdmax_startindex - 2)
    ddt_vn_adv = where(
        k_field >= k_lower & k_field < (nlev - 4),
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
        ),
        ddt_vn_adv,
    )

    return ddt_vn_adv


@program(backend=gtfn_cpu.run_gtfn)
def fused_stencils_19_to_20(
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    local_z_ekinh: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
    z_w_con_c_full: Field[[CellDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    zeta: Field[[VertexDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    ddt_vn_adv: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    levelmask: Field[[KDim], bool],
    k_field: Field[[KDim], int],
    scalfac_exdiff: float,
    area_edge: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    edge_startindex_nudging_plus_one: int,
    edge_endindex_interior: int,
    nrdmax_startindex: int,
    nlev: int,
):
    _fused_stencils_19_to_20(
        vt,
        vn_ie,
        z_kin_hor_e,
        local_z_ekinh,
        cfl_w_limit,
        dtime,
        z_w_con_c_full,
        coeff_gradekin,
        zeta,
        f_e,
        c_lin_e,
        ddqz_z_full_e,
        ddt_vn_adv,
        vn,
        inv_primal_edge_length,
        tangent_orientation,
        levelmask,
        k_field,
        scalfac_exdiff,
        area_edge,
        geofac_grdiv,
        nrdmax_startindex,
        nlev,
        domain={
            EdgeDim: (edge_startindex_nudging_plus_one, edge_endindex_interior),
        },
    )
