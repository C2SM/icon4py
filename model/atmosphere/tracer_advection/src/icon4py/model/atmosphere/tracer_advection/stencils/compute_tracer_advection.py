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
from gt4py.next import broadcast, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.tracer_advection.stencils.apply_density_increment import (
    _apply_density_increment,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    _apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_2nd_order_miura_horizontal_flux import (
    _compute_2nd_order_miura_horizontal_flux,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_upwind import (
    _compute_horizontal_tracer_flux_upwind,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    _compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm import _compute_ppm_flux
from icon4py.model.atmosphere.tracer_advection.stencils.compute_vertical_tracer_flux_upwind import (
    _compute_vertical_tracer_flux_upwind,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_horizontally import (
    _integrate_tracer_horizontally,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_vertically import (
    _integrate_tracer_vertically,
)
from icon4py.model.atmosphere.tracer_advection.tracer_advection_type import (
    HorizontalAdvectionLimiter,
    HorizontalAdvectionType,
    VerticalAdvectionType,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_vertical_tracer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    k: fa.KHalfField[gtx.int32],
    start_lvl: gtx.int32,
    start_lvl_plus1: gtx.int32,
    end_lvl: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    itype_vlimit: gtx.int32,
    ivadv_tracer: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    # Always zero (k >= 0). The dead `p_cc(KHalfDim + 0.5)` read is intentional: when
    # `ivadv_tracer` is baked in as NO_ADVECTION, it keeps the full-level K dimension
    # referenced in the program body; without it gtfn fails to compile the bindings
    # ("'K_t' is not a member of 'generated'"). Do not simplify to a plain broadcast.
    zero_flux = where(
        k < 0,
        p_cc(dims.KHalfDim + 0.5),
        broadcast(wpfloat(0.0), (dims.CellDim, dims.KHalfDim)),
    )
    return (
        # zero flux at the model top and bottom half levels
        concat_where(
            (dims.KHalfDim > 0) & (dims.KHalfDim < end_lvl + 1),
            _compute_vertical_tracer_flux_upwind(
                p_cc=p_cc,
                p_mflx_contra_v=p_mflx_contra_v,
            ),
            broadcast(wpfloat(0.0), (dims.CellDim, dims.KHalfDim)),
        )
        if (ivadv_tracer == VerticalAdvectionType.FIRST_ORDER_UPWIND)
        else (
            _compute_ppm_flux(
                p_cc=p_cc,
                p_cellmass_now=p_cellmass_now,
                p_mflx_contra_v=p_mflx_contra_v,
                p_cellhgt_mc_now=p_cellhgt_mc_now,
                k=k,
                start_lvl=start_lvl,
                start_lvl_plus1=start_lvl_plus1,
                end_lvl=end_lvl,
                wp_eps=wp_eps,
                p_dtime=p_dtime,
                itype_vlimit=itype_vlimit,
            )
            if (ivadv_tracer == VerticalAdvectionType.THIRD_ORDER_PPM)
            else zero_flux
        )
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_tracer_flux(
    p_upflux: fa.CellKHalfField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    k: fa.KHalfField[gtx.int32],
    start_lvl: gtx.int32,
    start_lvl_plus1: gtx.int32,
    end_lvl: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    itype_vlimit: gtx.int32,
    ivadv_tracer: gtx.int32,
    start_cell: gtx.int32,
    end_cell: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_tracer_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        k=k,
        start_lvl=start_lvl,
        start_lvl_plus1=start_lvl_plus1,
        end_lvl=end_lvl,
        wp_eps=wp_eps,
        p_dtime=p_dtime,
        itype_vlimit=itype_vlimit,
        ivadv_tracer=ivadv_tracer,
        out=p_upflux,
        domain={
            dims.CellDim: (start_cell, end_cell),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_tracer_advection_before_horizontal_limiter(
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    ivadv_tracer: gtx.int32,
    ihadv_tracer: gtx.int32,
    itype_hlimit: gtx.int32,
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
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    do_vertical_first: bool,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # p_mflx_tracer_v is pre-computed by the caller via compute_vertical_tracer_flux
    # and only read when do_vertical_first is set (even timesteps); on odd timesteps
    # the vertical flux is computed after the horizontal step.
    # It is NOT re-computed here to avoid fusing the large PPM kernel (which uses
    # concat_where on KHalfDim with fractional K-shifts) into this already-large
    # field_operator, which causes CUDA shared-memory overflow on aarch64/Santis.

    rhodz_ast2 = _apply_density_increment(
        rhodz_in=rhodz_now if do_vertical_first else rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        p_dtime=p_dtime,
        even_timestep=do_vertical_first,
    )

    p_tracer_after_vertical = (
        _integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_new=rhodz_ast2,
            p_dtime=p_dtime,
            ivadv_tracer=ivadv_tracer,
        )
        if do_vertical_first
        else p_tracer_now
    )

    tracer_for_h = p_tracer_after_vertical if do_vertical_first else p_tracer_now
    p_mflx_tracer_h_unlimited = (
        _compute_horizontal_tracer_flux_upwind(
            p_cc=tracer_for_h,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
        )
        if (ihadv_tracer == HorizontalAdvectionType.FIRST_ORDER_UPWIND)
        else (
            _compute_2nd_order_miura_horizontal_flux(
                p_cc=tracer_for_h,
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
            if (ihadv_tracer == HorizontalAdvectionType.SECOND_ORDER_LINEAR_MIURA)
            else broadcast(wpfloat(0.0), (dims.EdgeDim, dims.KDim))
        )
    )

    r_m = (
        _compute_positive_definite_horizontal_multiplicative_flux_factor(
            geofac_div=geofac_div,
            p_cc=tracer_for_h,
            p_rhodz_now=rhodz_ast2 if do_vertical_first else rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h_unlimited,
            p_dtime=p_dtime,
            wp_eps=wp_eps,
        )
        if (itype_hlimit == HorizontalAdvectionLimiter.POSITIVE_DEFINITE)
        else broadcast(wpfloat(1.0), (dims.CellDim, dims.KDim))
    )

    return (
        rhodz_ast2,
        p_tracer_after_vertical,
        p_mflx_tracer_h_unlimited,
        r_m,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_before_horizontal_limiter(
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    ivadv_tracer: gtx.int32,
    ihadv_tracer: gtx.int32,
    itype_hlimit: gtx.int32,
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
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    do_vertical_first: bool,
    start_cell_lateral_boundary_level_2: gtx.int32,
    end_cell_local: gtx.int32,
    end_cell_end: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_before_horizontal_limiter(
        rhodz_now=rhodz_now,
        rhodz_new=rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        p_tracer_now=p_tracer_now,
        p_mflx_tracer_v=p_mflx_tracer_v,
        p_mass_flx_e=p_mass_flx_e,
        p_vn=p_vn,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        ivadv_tracer=ivadv_tracer,
        ihadv_tracer=ihadv_tracer,
        itype_hlimit=itype_hlimit,
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
        wp_eps=wp_eps,
        p_dtime=p_dtime,
        do_vertical_first=do_vertical_first,
        out=(
            rhodz_ast2,
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
def _compute_horizontal_tracer_advection(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    end_lvl: gtx.int32,
    ihadv_tracer: gtx.int32,
    itype_hlimit: gtx.int32,
    p_dtime: ta.wpfloat,
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """Apply the horizontal flux limiter and integrate the tracer horizontally.

    Returns the limited horizontal tracer flux and the tracer after horizontal transport.
    """
    p_mflx_tracer_h_new = (
        _apply_positive_definite_horizontal_multiplicative_flux_factor(
            r_m=r_m, p_mflx_tracer_h=p_mflx_tracer_h_unlimited
        )
        if (itype_hlimit == HorizontalAdvectionLimiter.POSITIVE_DEFINITE)
        else p_mflx_tracer_h_unlimited
    )
    p_tracer_after_horizontal = concat_where(
        (dims.KDim >= 0) & (dims.KDim < end_lvl + 1),
        (
            _integrate_tracer_horizontally(
                p_mflx_tracer_h=p_mflx_tracer_h_new,
                deepatmo_divh=deepatmo_divh,
                tracer_now=tracer_now,
                rhodz_now=rhodz_now,
                rhodz_new=rhodz_new,
                geofac_div=geofac_div,
                p_dtime=p_dtime,
            )
            if (ihadv_tracer != HorizontalAdvectionType.NO_ADVECTION)
            else tracer_now
        ),
        broadcast(wpfloat(0.0), (dims.CellDim, dims.KDim)),
    )
    return p_mflx_tracer_h_new, p_tracer_after_horizontal


@gtx.field_operator
def _compute_tracer_advection_after_horizontal_limiter(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    start_lvl: gtx.int32,
    start_lvl_plus1: gtx.int32,
    end_lvl: gtx.int32,
    do_vertical_first: bool,
    ivadv_tracer: gtx.int32,
    ihadv_tracer: gtx.int32,
    itype_hlimit: gtx.int32,
    itype_vlimit: gtx.int32,
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.CellKHalfField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    tracer_now_for_h = p_tracer_after_vertical if do_vertical_first else p_tracer_now
    rhodz_for_h_now = rhodz_ast2 if do_vertical_first else rhodz_now
    rhodz_for_h_new = rhodz_new if do_vertical_first else rhodz_ast2

    p_mflx_tracer_h_new, p_tracer_after_horizontal = _compute_horizontal_tracer_advection(
        r_m=r_m,
        p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
        tracer_now=tracer_now_for_h,
        rhodz_now=rhodz_for_h_now,
        rhodz_new=rhodz_for_h_new,
        deepatmo_divh=deepatmo_divh,
        geofac_div=geofac_div,
        end_lvl=end_lvl,
        ihadv_tracer=ihadv_tracer,
        itype_hlimit=itype_hlimit,
        p_dtime=p_dtime,
    )
    p_mflx_tracer_v_computed = _compute_vertical_tracer_flux(
        p_cc=p_tracer_after_horizontal,
        p_cellmass_now=rhodz_ast2,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        k=k_half,
        start_lvl=start_lvl,
        start_lvl_plus1=start_lvl_plus1,
        end_lvl=end_lvl,
        wp_eps=wp_eps,
        p_dtime=p_dtime,
        itype_vlimit=itype_vlimit,
        ivadv_tracer=ivadv_tracer,
    )
    p_mflx_tracer_v_out = p_mflx_tracer_v if do_vertical_first else p_mflx_tracer_v_computed
    p_tracer_new = (
        p_tracer_after_horizontal
        if do_vertical_first
        else _integrate_tracer_vertically(
            tracer_now=p_tracer_after_horizontal,
            rhodz_now=rhodz_ast2,
            p_mflx_tracer_v=p_mflx_tracer_v_computed,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_new=rhodz_new,
            p_dtime=p_dtime,
            ivadv_tracer=ivadv_tracer,
        )
    )
    return p_mflx_tracer_h_new, p_mflx_tracer_v_out, p_tracer_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tracer_advection_after_horizontal_limiter(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    p_tracer_new: fa.CellKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h_unlimited: fa.EdgeKField[ta.wpfloat],
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_tracer_after_vertical: fa.CellKField[ta.wpfloat],
    rhodz_ast2: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    start_lvl: gtx.int32,
    start_lvl_plus1: gtx.int32,
    end_lvl: gtx.int32,
    do_vertical_first: bool,
    ivadv_tracer: gtx.int32,
    ihadv_tracer: gtx.int32,
    itype_hlimit: gtx.int32,
    itype_vlimit: gtx.int32,
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    start_cell_nudging: gtx.int32,
    end_cell_local: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tracer_advection_after_horizontal_limiter(
        r_m=r_m,
        p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
        p_tracer_now=p_tracer_now,
        p_tracer_after_vertical=p_tracer_after_vertical,
        p_mflx_tracer_v=p_mflx_tracer_v,
        rhodz_ast2=rhodz_ast2,
        rhodz_now=rhodz_now,
        rhodz_new=rhodz_new,
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        deepatmo_divh=deepatmo_divh,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        k_half=k_half,
        start_lvl=start_lvl,
        start_lvl_plus1=start_lvl_plus1,
        end_lvl=end_lvl,
        do_vertical_first=do_vertical_first,
        ivadv_tracer=ivadv_tracer,
        ihadv_tracer=ihadv_tracer,
        itype_hlimit=itype_hlimit,
        itype_vlimit=itype_vlimit,
        geofac_div=geofac_div,
        wp_eps=wp_eps,
        p_dtime=p_dtime,
        out=(p_mflx_tracer_h, p_mflx_tracer_v, p_tracer_new),
        domain=(
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo),
                dims.KDim: (0, vertical_end),
            },
            {
                dims.CellDim: (start_cell_nudging, end_cell_local),
                dims.KHalfDim: (0, vertical_end + 1),
            },
            {
                dims.CellDim: (start_cell_nudging, end_cell_local),
                dims.KDim: (0, vertical_end),
            },
        ),
    )
