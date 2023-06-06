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

# flake8: noqa

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import program
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.calculate_nabla2_and_smag_coefficients_for_vn import (
    _calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_02_03 import (
    _fused_mo_nh_diffusion_stencil_02_03,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_04_05_06 import (
    _fused_mo_nh_diffusion_stencil_04_05_06,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_07_08_09_10 import (
    _fused_mo_nh_diffusion_stencil_07_08_09_10,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_11_12 import (
    _fused_mo_nh_diffusion_stencil_11_12,
)
from icon4py.atm_dyn_iconam.fused_mo_nh_diffusion_stencil_13_14 import (
    _fused_mo_nh_diffusion_stencil_13_14,
)
from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_15 import (
    _mo_nh_diffusion_stencil_15,
)
from icon4py.atm_dyn_iconam.update_theta_and_exner import _update_theta_and_exner
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CECDim,
    CellDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.utils import _scale_k, _set_zero_v_k


@program
def diffusion_run(
    diagnostic_hdef_ic: Field[[CellDim, KDim], float],
    diagnostic_div_ic: Field[[CellDim, KDim], float],
    diagnostic_dwdx: Field[[CellDim, KDim], float],
    diagnostic_dwdy: Field[[CellDim, KDim], float],
    prognostic_w: Field[[CellDim, KDim], float],
    prognostic_vn: Field[[EdgeDim, KDim], float],
    prognostic_exner_pressure: Field[[CellDim, KDim], float],
    prognostic_theta_v: Field[[CellDim, KDim], float],
    metric_theta_ref_mc: Field[[CellDim, KDim], float],
    metric_wgtfac_c: Field[[CellDim, KDim], float],
    metric_mask_hdiff: Field[[CellDim, KDim], bool],
    metric_zd_vertidx: Field[[CECDim, KDim], int32],
    metric_zd_diffcoef: Field[[CellDim, KDim], float],
    metric_zd_intcoef: Field[[CECDim, KDim], float],
    interpolation_e_bln_c_s: Field[[CellDim, C2EDim], float],
    interpolation_rbf_coeff_1: Field[[VertexDim, V2EDim], float],
    interpolation_rbf_coeff_2: Field[[VertexDim, V2EDim], float],
    interpolation_geofac_div: Field[[CellDim, C2EDim], float],
    interpolation_geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    interpolation_geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    interpolation_nudgecoeff_e: Field[[EdgeDim], float],
    interpolation_geofac_n2s: Field[[CellDim, C2E2CODim], float],
    interpolation_geofac_n2s_c: Field[[CellDim], float],
    interpolation_geofac_n2s_nbh: Field[[CECDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inverse_primal_edge_lengths: Field[[EdgeDim], float],
    inverse_dual_edge_lengths: Field[[EdgeDim], float],
    inverse_vert_vert_lengths: Field[[EdgeDim], float],
    primal_normal_vert_1: Field[[ECVDim], float],
    primal_normal_vert_2: Field[[ECVDim], float],
    dual_normal_vert_1: Field[[ECVDim], float],
    dual_normal_vert_2: Field[[ECVDim], float],
    edge_areas: Field[[EdgeDim], float],
    cell_areas: Field[[CellDim], float],
    diff_multfac_vn: Field[[KDim], float],
    dtime: float,
    rd_o_cvd: float,
    local_thresh_tdiff: float,
    local_smag_limit: Field[[KDim], float],
    local_u_vert: Field[[VertexDim, KDim], float],
    local_v_vert: Field[[VertexDim, KDim], float],
    local_enh_smag_fac: Field[[KDim], float],
    local_kh_smag_e: Field[[EdgeDim, KDim], float],
    local_kh_smag_ec: Field[[EdgeDim, KDim], float],
    local_z_nabla2_e: Field[[EdgeDim, KDim], float],
    local_z_temp: Field[[CellDim, KDim], float],
    local_diff_multfac_smag: Field[[KDim], float],
    local_diff_multfac_n2w: Field[[KDim], float],
    local_smag_offset: float,
    local_nudgezone_diff: float,
    local_fac_bdydiff_v: float,
    local_diff_multfac_w: float,
    local_vertical_index: Field[[KDim], int32],
    local_horizontal_cell_index: Field[[CellDim], int32],
    local_horizontal_edge_index: Field[[EdgeDim], int32],
    cell_startindex_interior: int32,
    cell_halo_idx: int32,
    cell_startindex_nudging: int,
    cell_endindex_local_plus1: int,
    cell_endindex_local: int,
    edge_startindex_nudging_plus1: int,
    edge_startindex_nudging_minus1: int32,
    edge_endindex_local: int,
    edge_endindex_local_minus2: int,
    vertex_startindex_lb_plus3: int,
    vertex_startindex_lb_plus1: int,
    vertex_endindex_local: int,
    vertex_endindex_local_minus1: int,
    index_of_damping_height: int32,
    nlev: int,
    boundary_diffusion_start_index_edges: int,
):
    _scale_k(local_enh_smag_fac, dtime, out=local_diff_multfac_smag)

    # TODO: @magdalena is this needed?, if not remove
    _set_zero_v_k(out=local_u_vert)
    _set_zero_v_k(out=local_v_vert)

    # # 1.  CALL rbf_vec_interpol_vertex
    _mo_intp_rbf_rbf_vec_interpol_vertex(
        prognostic_vn,
        interpolation_rbf_coeff_1,
        interpolation_rbf_coeff_2,
        out=(local_u_vert, local_v_vert),
        domain={
            VertexDim: (vertex_startindex_lb_plus1, vertex_endindex_local_minus1),
            KDim: (0, nlev),
        },
    )

    # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult
    # 3.  mo_nh_diffusion_stencil_01, mo_nh_diffusion_stencil_02, mo_nh_diffusion_stencil_03

    _calculate_nabla2_and_smag_coefficients_for_vn(
        local_diff_multfac_smag,
        tangent_orientation,
        inverse_primal_edge_lengths,
        inverse_vert_vert_lengths,
        local_u_vert,
        local_v_vert,
        primal_normal_vert_1,
        primal_normal_vert_2,
        dual_normal_vert_1,
        dual_normal_vert_2,
        prognostic_vn,
        local_smag_limit,
        local_smag_offset,
        out=(local_kh_smag_e, local_kh_smag_ec, local_z_nabla2_e),
        domain={
            # TODO @magdalena wrong start index??
            EdgeDim: (boundary_diffusion_start_index_edges, edge_endindex_local_minus2),
            KDim: (0, nlev),
        },
    )

    _fused_mo_nh_diffusion_stencil_02_03(
        local_kh_smag_ec,
        prognostic_vn,
        interpolation_e_bln_c_s,
        interpolation_geofac_div,
        local_diff_multfac_smag,
        metric_wgtfac_c,
        out=(
            diagnostic_div_ic,
            diagnostic_hdef_ic,
        ),
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (1, nlev),
        },
    )
    #
    # # 4.  IF (discr_vn > 1) THEN CALL sync_patch_array -> false for MCH
    #
    # # 5.  CALL rbf_vec_interpol_vertex_wp
    _mo_intp_rbf_rbf_vec_interpol_vertex(
        local_z_nabla2_e,
        interpolation_rbf_coeff_1,
        interpolation_rbf_coeff_2,
        out=(local_u_vert, local_v_vert),
        domain={
            VertexDim: (vertex_startindex_lb_plus3, vertex_endindex_local),
            KDim: (0, nlev),
        },
    )
    # # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult
    #
    # # 7.  mo_nh_diffusion_stencil_04, mo_nh_diffusion_stencil_05
    # # 7a. IF (l_limited_area .OR. jg > 1) mo_nh_diffusion_stencil_06
    #
    _fused_mo_nh_diffusion_stencil_04_05_06(
        local_u_vert,
        local_v_vert,
        primal_normal_vert_1,
        primal_normal_vert_2,
        local_z_nabla2_e,
        inverse_vert_vert_lengths,
        inverse_primal_edge_lengths,
        edge_areas,
        local_kh_smag_e,
        diff_multfac_vn,
        interpolation_nudgecoeff_e,
        prognostic_vn,
        local_horizontal_edge_index,
        local_nudgezone_diff,
        local_fac_bdydiff_v,
        edge_startindex_nudging_minus1,
        out=prognostic_vn,
        domain={
            EdgeDim: (edge_startindex_nudging_plus1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )
    # # 7b. mo_nh_diffusion_stencil_07, mo_nh_diffusion_stencil_08,
    # #     mo_nh_diffusion_stencil_09, mo_nh_diffusion_stencil_10

    _fused_mo_nh_diffusion_stencil_07_08_09_10(
        cell_areas,
        interpolation_geofac_n2s,
        interpolation_geofac_grg_x,
        interpolation_geofac_grg_y,
        prognostic_w,
        prognostic_w,
        diagnostic_dwdx,
        diagnostic_dwdy,
        local_diff_multfac_w,
        local_diff_multfac_n2w,
        local_vertical_index,
        local_horizontal_cell_index,
        index_of_damping_height,
        cell_startindex_interior,
        cell_halo_idx,
        out=(
            prognostic_w,
            diagnostic_dwdx,
            diagnostic_dwdy,
        ),
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local_plus1),
            KDim: (0, nlev),
        },
    )
    # # 8.  HALO EXCHANGE: CALL sync_patch_array
    # # 9.  mo_nh_diffusion_stencil_11, mo_nh_diffusion_stencil_12, mo_nh_diffusion_stencil_13,
    # #     mo_nh_diffusion_stencil_14, mo_nh_diffusion_stencil_15, mo_nh_diffusion_stencil_16
    #
    # # TODO @magdalena check: kh_smag_e is an out field, should  not be calculated in init?
    #

    _fused_mo_nh_diffusion_stencil_11_12(
        prognostic_theta_v,
        metric_theta_ref_mc,
        local_thresh_tdiff,
        local_kh_smag_e,
        out=local_kh_smag_e,
        domain={
            EdgeDim: (edge_startindex_nudging_plus1, edge_endindex_local),
            KDim: (nlev - 2, nlev),
        },
    )
    _fused_mo_nh_diffusion_stencil_13_14(
        local_kh_smag_e,
        inverse_dual_edge_lengths,
        prognostic_theta_v,
        interpolation_geofac_div,
        out=local_z_temp,
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (0, nlev),
        },
    )

    # MO_NH_DIFFUSION_STENCIL_15: as_offset index fields!
    _mo_nh_diffusion_stencil_15(
        metric_mask_hdiff,
        metric_zd_vertidx,
        metric_zd_diffcoef,
        interpolation_geofac_n2s_c,
        interpolation_geofac_n2s_nbh,
        metric_zd_intcoef,
        prognostic_theta_v,
        local_z_temp,
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (0, nlev),
        },
        out=local_z_temp,
    )

    _update_theta_and_exner(
        local_z_temp,
        cell_areas,
        prognostic_theta_v,
        prognostic_exner_pressure,
        rd_o_cvd,
        out=(prognostic_theta_v, prognostic_exner_pressure),
        domain={
            CellDim: (cell_startindex_nudging, cell_endindex_local),
            KDim: (0, nlev),
        },
    )
    # 10. HALO EXCHANGE sync_patch_array
