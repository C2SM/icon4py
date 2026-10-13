# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, maximum, minimum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    _apply_diffusion_to_theta_and_exner,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import _apply_diffusion_to_vn
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    _calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _limit_smag_coefficients(
    kh_smag: fa.EdgeKField[vpfloat],
    smag_limit: fa.KField[vpfloat],
    smag_offset: vpfloat,
) -> fa.EdgeKField[vpfloat]:
    """Limit the raw Smagorinsky diffusion coefficient to the interval [0, smag_limit]."""
    kh_smag_wp, smag_offset_wp = astype((kh_smag, smag_offset), wpfloat)
    kh_smag_limited = maximum(vpfloat("0.0"), astype(kh_smag_wp - smag_offset_wp, vpfloat))
    kh_smag_limited = minimum(kh_smag_limited, smag_limit)
    return kh_smag_limited


@gtx.field_operator
def _diffuse_prognostic_fields(
    kh_smag: fa.EdgeKField[vpfloat],
    smag_limit: fa.KField[vpfloat],
    smag_offset: vpfloat,
    vn_in: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    compute_diagnostic_quantities: bool,
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: gtx.int32,
    limited_area: bool,
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKHalfField[vpfloat],
    dwdy: fa.CellKHalfField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    theta_v_in: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
    apply_to_temperature: bool,
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    """Fused diffusion of the prognostic fields vn, w, theta_v and exner.

    Composes the Smagorinsky limiter, the turbulence diagnostics (from pre-diffusion
    vn), the diffusion of the normal wind, the diffusion of the vertical wind with the
    horizontal wind gradients for turbulence, and the diffusion of theta_v and exner
    with the grid-point cold-pool enhancement of the diffusion coefficient inlined.

    Note:
        The cold-pool enhancement is no longer published through `kh_smag_e`. If ICON's
        moisture diffusion (`lhdiff_q`, which consumes the enhanced coefficient) is ever
        ported, the enhanced coefficient must be published again.
    """
    kh_smag_limited = _limit_smag_coefficients(kh_smag, smag_limit, smag_offset)

    div_ic, hdef_ic = (
        _calculate_diagnostic_quantities_for_turbulence(
            kh_smag=kh_smag,
            vn=vn_in,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            wgtfac_c=wgtfac_c,
        )
        if compute_diagnostic_quantities
        else (div_ic, hdef_ic)
    )

    vn = _apply_diffusion_to_vn(
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_v1=primal_normal_vert_v1,
        primal_normal_vert_v2=primal_normal_vert_v2,
        z_nabla2_e=z_nabla2_e,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_primal_edge_length=inv_primal_edge_length,
        area_edge=area_edge,
        kh_smag_e=kh_smag_limited,
        diff_multfac_vn=diff_multfac_vn,
        nudgecoeff_e=nudgecoeff_e,
        vn=vn_in,
        nudgezone_diff=nudgezone_diff,
        fac_bdydiff_v=fac_bdydiff_v,
        start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
        limited_area=limited_area,
    )

    w, dwdx, dwdy = _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
        area=area,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        w_old=w_old,
        type_shear=type_shear,
        dwdx=dwdx,
        dwdy=dwdy,
        diff_multfac_w=diff_multfac_w,
        diff_multfac_n2w=diff_multfac_n2w,
        nrdmax=nrdmax,
        interior_idx=interior_idx,
        halo_idx=halo_idx,
    )

    # TODO(havogt): Note that `concat_where(dims.KDim == nlev-2, ...)` is currently broken
    # because of insufficiency in the domain inference of GT4Py,
    # see https://github.com/GridTools/gt4py/issues/2205.
    theta_v, exner = (
        _apply_diffusion_to_theta_and_exner(
            kh_smag_e=concat_where(
                dims.KDim >= nlev - 2,
                _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
                    theta_v=theta_v_in,
                    theta_ref_mc=theta_ref_mc,
                    thresh_tdiff=thresh_tdiff,
                    smallest_vpfloat=smallest_vpfloat,
                    kh_smag_e=kh_smag_limited,
                ),
                kh_smag_limited,
            ),
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v_in,
            geofac_div=geofac_div,
            zd_vertoffset=zd_vertoffset,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh,
            vcoef=vcoef,
            area=area,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            apply_zdiffusion_t=apply_zdiffusion_t,
        )
        if apply_to_temperature
        else (theta_v, exner)
    )

    return vn, w, dwdx, dwdy, theta_v, exner, div_ic, hdef_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diffuse_prognostic_fields(
    kh_smag: fa.EdgeKField[vpfloat],
    smag_limit: fa.KField[vpfloat],
    smag_offset: vpfloat,
    vn_in: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    area_edge: fa.EdgeField[wpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKHalfField[vpfloat],
    dwdy: fa.CellKHalfField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    theta_v_in: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
    compute_diagnostic_quantities: bool,
    apply_to_temperature: bool,
    limited_area: bool,
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    start_2nd_nudge_line_idx_e: gtx.int32,
    nrdmax: gtx.int32,
    nlev: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    edge_horizontal_start: gtx.int32,
    edge_horizontal_end: gtx.int32,
    w_horizontal_start: gtx.int32,
    w_horizontal_end: gtx.int32,
    cell_horizontal_start: gtx.int32,
    cell_horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _diffuse_prognostic_fields(
        kh_smag=kh_smag,
        smag_limit=smag_limit,
        smag_offset=smag_offset,
        vn_in=vn_in,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        diff_multfac_smag=diff_multfac_smag,
        wgtfac_c=wgtfac_c,
        div_ic=div_ic,
        hdef_ic=hdef_ic,
        compute_diagnostic_quantities=compute_diagnostic_quantities,
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_v1=primal_normal_vert_v1,
        primal_normal_vert_v2=primal_normal_vert_v2,
        z_nabla2_e=z_nabla2_e,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_primal_edge_length=inv_primal_edge_length,
        area_edge=area_edge,
        diff_multfac_vn=diff_multfac_vn,
        nudgecoeff_e=nudgecoeff_e,
        nudgezone_diff=nudgezone_diff,
        fac_bdydiff_v=fac_bdydiff_v,
        start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
        limited_area=limited_area,
        area=area,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        w_old=w_old,
        type_shear=type_shear,
        dwdx=dwdx,
        dwdy=dwdy,
        diff_multfac_w=diff_multfac_w,
        diff_multfac_n2w=diff_multfac_n2w,
        nrdmax=nrdmax,
        interior_idx=interior_idx,
        halo_idx=halo_idx,
        theta_v_in=theta_v_in,
        inv_dual_edge_length=inv_dual_edge_length,
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
        geofac_n2s_c=geofac_n2s_c,
        geofac_n2s_nbh=geofac_n2s_nbh,
        vcoef=vcoef,
        theta_v=theta_v,
        exner=exner,
        theta_ref_mc=theta_ref_mc,
        thresh_tdiff=thresh_tdiff,
        smallest_vpfloat=smallest_vpfloat,
        rd_o_cvd=rd_o_cvd,
        apply_zdiffusion_t=apply_zdiffusion_t,
        apply_to_temperature=apply_to_temperature,
        nlev=nlev,
        out=(vn, w, dwdx, dwdy, theta_v, exner, div_ic, hdef_ic),
        domain=(
            {
                dims.EdgeDim: (edge_horizontal_start, edge_horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (interior_idx, halo_idx),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (w_horizontal_start, w_horizontal_end),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (w_horizontal_start, w_horizontal_end),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
        ),
    )
