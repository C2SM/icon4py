# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Fused programs for the production tracer advection configuration.

The horizontal positive-definite limiter requires a cell exchange between
computing its factor and applying it. The programs intentionally preserve
that boundary.
"""

import gt4py.next as gtx
from gt4py.next import broadcast
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.tracer_advection.stencils.apply_density_increment import (
    _apply_density_increment,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    _apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_barycentric_backtrajectory_alt import (
    _compute_barycentric_backtrajectory_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    _compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    _compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm4gpu_courant_number import (
    _compute_ppm4gpu_courant_number,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm4gpu_fractional_flux import (
    _compute_ppm4gpu_fractional_flux,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm4gpu_integer_flux import (
    _compute_ppm4gpu_integer_flux,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    _compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_quadratic_face_values import (
    _compute_ppm_quadratic_face_values,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_quartic_face_values import (
    _compute_ppm_quartic_face_values,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_slope import _compute_ppm_slope
from icon4py.model.atmosphere.tracer_advection.stencils.compute_vertical_parabola_limiter_condition import (
    _compute_vertical_parabola_limiter_condition,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_horizontally import (
    _integrate_tracer_horizontally,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_vertically import (
    _integrate_tracer_vertically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.limit_vertical_parabola_semi_monotonically import (
    _limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.limit_vertical_slope_semi_monotonically import (
    _limit_vertical_slope_semi_monotonically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    _reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_wp,
)


@gtx.field_operator
def _compute_ppm4gpu_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    z_cfl = broadcast(0.0, (dims.CellDim, dims.KHalfDim))
    z_cfl = concat_where(
        (dims.KDim > 0) & (dims.KDim < elev + 1),
        _compute_ppm4gpu_courant_number(
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellmass_now=p_cellmass_now,
            z_cfl=z_cfl,
            k_half=k,
            slevp1_ti=slevp1_ti,
            nlev=elev,
            dbl_eps=dbl_eps,
            p_dtime=p_dtime,
        ),
        z_cfl,
    )
    z_slope = _limit_vertical_slope_semi_monotonically(
        p_cc=p_cc,
        z_slope=_compute_ppm_slope(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, elev=elev),
        k=k,
        elev=elev,
    )
    p_face = concat_where(
        (dims.KDim > 1) & (dims.KDim < elev),
        _compute_ppm_quartic_face_values(
            p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, z_slope=z_slope
        ),
        _compute_ppm_quadratic_face_values(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
    )
    p_face = concat_where(dims.KDim > 0, p_face, p_cc)
    p_face = concat_where(dims.KDim < elev + 1, p_face, p_cc(dims.KDim - 1))
    l_limit = _compute_vertical_parabola_limiter_condition(p_face=p_face, p_cc=p_cc)
    z_face_up, z_face_low = _limit_vertical_parabola_semi_monotonically(
        l_limit=l_limit, p_face=p_face, p_cc=p_cc
    )
    z_delta_q, z_a1 = _compute_ppm4gpu_parabola_coefficients(
        z_face_up=z_face_up, z_face_low=z_face_low, p_cc=p_cc
    )
    p_upflux = _compute_ppm4gpu_fractional_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        z_delta_q=z_delta_q,
        z_a1=z_a1,
        k_half=k,
        slev=slev,
        p_dtime=p_dtime,
    )
    p_upflux = _compute_ppm4gpu_integer_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        p_upflux=p_upflux,
        k_half=k,
        slev=slev,
        p_dtime=p_dtime,
    )
    return concat_where(
        (dims.KDim > 0) & (dims.KDim < elev + 1),
        p_upflux,
        broadcast(0.0, (dims.CellDim, dims.KDim)),
    )


@gtx.field_operator
def _compute_unlimited_horizontal_tracer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.EdgeKField[ta.wpfloat]:
    z_real_vt = _compute_tangential_wind_wp(vn=p_vn, rbf_vec_coeff_e=rbf_vec_coeff_e)
    p_distv_bary_1, p_distv_bary_2 = _compute_barycentric_backtrajectory_alt(
        p_vn=p_vn,
        p_vt=z_real_vt,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        p_dthalf=0.5 * p_dtime,
    )
    p_coeff_1, p_coeff_2, p_coeff_3 = _reconstruct_linear_coefficients_svd(
        p_cc=p_cc, lsq_pseudoinv_1=lsq_pseudoinv_1, lsq_pseudoinv_2=lsq_pseudoinv_2
    )
    return _compute_horizontal_tracer_flux_from_linear_coefficients_alt(
        z_lsq_coeff_1=p_coeff_1,
        z_lsq_coeff_2=p_coeff_2,
        z_lsq_coeff_3=p_coeff_3,
        distv_bary_1=p_distv_bary_1,
        distv_bary_2=p_distv_bary_2,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
    )


@gtx.field_operator
def _compute_tracer_advection_even_timestep_before_horizontal_limiter(
    rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    rhodz_ast2 = _apply_density_increment(
        rhodz_in=rhodz_now,
        p_mflx_contra_v=p_mflx_contra_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        p_dtime=p_dtime,
        even_timestep=True,
    )
    p_mflx_tracer_v = _compute_ppm4gpu_flux(
        p_cc=p_tracer_now,
        p_cellmass_now=rhodz_now,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        k=k,
        slev=slev,
        slevp1_ti=slevp1_ti,
        elev=elev,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
    )
    p_tracer_after_vertical = _integrate_tracer_vertically(
        tracer_now=p_tracer_now,
        rhodz_now=rhodz_now,
        p_mflx_tracer_v=p_mflx_tracer_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        rhodz_new=rhodz_ast2,
        k=k,
        p_dtime=p_dtime,
        ivadv_tracer=ivadv_tracer,
        iadv_slev_jt=iadv_slev_jt,
    )
    p_mflx_tracer_h_unlimited = _compute_unlimited_horizontal_tracer_flux(
        p_cc=p_tracer_after_vertical,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        p_dtime=p_dtime,
    )
    r_m = _compute_positive_definite_horizontal_multiplicative_flux_factor(
        geofac_div=geofac_div,
        p_cc=p_tracer_after_vertical,
        p_rhodz_now=rhodz_ast2,
        p_mflx_tracer_h=p_mflx_tracer_h_unlimited,
        p_dtime=p_dtime,
        dbl_eps=dbl_eps,
    )
    return (
        rhodz_ast2,
        p_mflx_tracer_v,
        p_tracer_after_vertical,
        p_mflx_tracer_h_unlimited,
        r_m,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_even_timestep_before_horizontal_limiter(
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    start_cell_lateral_boundary_level_2: gtx.int32,
    end_cell_local: gtx.int32,
    end_cell_end: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_even_timestep_before_horizontal_limiter(
        rhodz_now=rhodz_now,
        p_mflx_contra_v=p_mflx_contra_v,
        p_tracer_now=p_tracer_now,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        k=k,
        slev=slev,
        slevp1_ti=slevp1_ti,
        elev=elev,
        ivadv_tracer=ivadv_tracer,
        iadv_slev_jt=iadv_slev_jt,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        geofac_div=geofac_div,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
        out=(
            rhodz_ast2,
            p_mflx_tracer_v,
            p_tracer_after_vertical,
            p_mflx_tracer_h_unlimited,
            r_m,
        ),
        domain=(
            {
                dims.CellDim: (start_cell_lateral_boundary_level_2, end_cell_end),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_2, end_cell_end),
                dims.KDim: (0, vertical_end + 1),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_2, end_cell_end),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_2, end_cell_local),
                dims.KDim: (0, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _compute_tracer_advection_even_timestep_after_horizontal_limiter(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    p_mflx_tracer_h = _apply_positive_definite_horizontal_multiplicative_flux_factor(
        r_m=r_m, p_mflx_tracer_h=p_mflx_tracer_h_unlimited
    )
    p_tracer_new = _integrate_tracer_horizontally(
        p_mflx_tracer_h=p_mflx_tracer_h,
        deepatmo_divh=deepatmo_divh,
        tracer_now=p_tracer_after_vertical,
        rhodz_now=rhodz_ast2,
        rhodz_new=rhodz_new,
        geofac_div=geofac_div,
        p_dtime=p_dtime,
    )
    return p_mflx_tracer_h, p_tracer_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_even_timestep_after_horizontal_limiter(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_tracer_new: fa.CellKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
    start_cell_nudging: gtx.int32,
    end_cell_local: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_even_timestep_after_horizontal_limiter(
        r_m=r_m,
        p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
        p_tracer_after_vertical=p_tracer_after_vertical,
        rhodz_ast2=rhodz_ast2,
        rhodz_new=rhodz_new,
        deepatmo_divh=deepatmo_divh,
        geofac_div=geofac_div,
        p_dtime=p_dtime,
        out=(p_mflx_tracer_h, p_tracer_new),
        domain=(
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_nudging, end_cell_local),
                dims.KDim: (0, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _compute_tracer_advection_odd_timestep_before_horizontal_limiter(
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    rhodz_ast2 = _apply_density_increment(
        rhodz_in=rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        p_dtime=p_dtime,
        even_timestep=False,
    )
    p_mflx_tracer_h_unlimited = _compute_unlimited_horizontal_tracer_flux(
        p_cc=p_tracer_now,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        p_dtime=p_dtime,
    )
    r_m = _compute_positive_definite_horizontal_multiplicative_flux_factor(
        geofac_div=geofac_div,
        p_cc=p_tracer_now,
        p_rhodz_now=rhodz_now,
        p_mflx_tracer_h=p_mflx_tracer_h_unlimited,
        p_dtime=p_dtime,
        dbl_eps=dbl_eps,
    )
    return rhodz_ast2, p_mflx_tracer_h_unlimited, r_m


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_odd_timestep_before_horizontal_limiter(
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    start_cell_lateral_boundary_level_2: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    end_cell_local: gtx.int32,
    end_cell_end: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_odd_timestep_before_horizontal_limiter(
        rhodz_now=rhodz_now,
        rhodz_new=rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        p_tracer_now=p_tracer_now,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        geofac_div=geofac_div,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
        out=(rhodz_ast2, p_mflx_tracer_h_unlimited, r_m),
        domain=(
            {
                dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_end),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_2, end_cell_local),
                dims.KDim: (0, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _compute_tracer_advection_odd_timestep_after_horizontal_limiter(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    # PPM accesses vertical neighbors; padding keeps its local input defined at inferred offsets.
    p_mflx_tracer_h = concat_where(
        (dims.KDim >= 0) & (dims.KDim < elev + 1),
        _apply_positive_definite_horizontal_multiplicative_flux_factor(
            r_m=r_m, p_mflx_tracer_h=p_mflx_tracer_h_unlimited
        ),
        broadcast(0.0, (dims.EdgeDim, dims.KDim)),
    )
    p_tracer_after_horizontal = concat_where(
        (dims.KDim >= 0) & (dims.KDim < elev + 1),
        _integrate_tracer_horizontally(
            p_mflx_tracer_h=p_mflx_tracer_h,
            deepatmo_divh=deepatmo_divh,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_ast2,
            geofac_div=geofac_div,
            p_dtime=p_dtime,
        ),
        broadcast(0.0, (dims.CellDim, dims.KDim)),
    )
    p_mflx_tracer_v = _compute_ppm4gpu_flux(
        p_cc=p_tracer_after_horizontal,
        p_cellmass_now=rhodz_ast2,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        k=k,
        slev=slev,
        slevp1_ti=slevp1_ti,
        elev=elev,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
    )
    p_tracer_new = _integrate_tracer_vertically(
        tracer_now=p_tracer_after_horizontal,
        rhodz_now=rhodz_ast2,
        p_mflx_tracer_v=p_mflx_tracer_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        rhodz_new=rhodz_new,
        k=k,
        p_dtime=p_dtime,
        ivadv_tracer=ivadv_tracer,
        iadv_slev_jt=iadv_slev_jt,
    )
    return p_mflx_tracer_h, p_mflx_tracer_v, p_tracer_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_odd_timestep_after_horizontal_limiter(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKField[ta.wpfloat],
    p_tracer_new: fa.CellKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    start_cell_nudging: gtx.int32,
    end_cell_local: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_odd_timestep_after_horizontal_limiter(
        r_m=r_m,
        p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
        p_tracer_now=p_tracer_now,
        rhodz_ast2=rhodz_ast2,
        rhodz_now=rhodz_now,
        rhodz_new=rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        deepatmo_divh=deepatmo_divh,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        k_half=k,
        slev=slev,
        slevp1_ti=slevp1_ti,
        elev=elev,
        ivadv_tracer=ivadv_tracer,
        iadv_slev_jt=iadv_slev_jt,
        geofac_div=geofac_div,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
        out=(p_mflx_tracer_h, p_mflx_tracer_v, p_tracer_new),
        domain=(
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_nudging, end_cell_local),
                dims.KDim: (0, vertical_end + 1),
            },
            {
                dims.CellDim: (start_cell_nudging, end_cell_local),
                dims.KDim: (0, vertical_end),
            },
        ),
    )
