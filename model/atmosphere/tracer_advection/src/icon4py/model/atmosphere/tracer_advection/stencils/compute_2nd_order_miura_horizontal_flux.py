# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Horizontal tracer flux with the second-order Miura scheme (linear reconstruction)."""

import gt4py.next as gtx

from icon4py.model.atmosphere.tracer_advection.stencils.compute_barycentric_backtrajectory_alt import (
    _compute_barycentric_backtrajectory_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    _compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    _reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_2nd_order_miura_horizontal_flux(
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
    z_real_vt = _compute_tangential_wind(vn=p_vn, rbf_vec_coeff_e=rbf_vec_coeff_e)
    p_distv_bary_1, p_distv_bary_2 = _compute_barycentric_backtrajectory_alt(
        p_vn=p_vn,
        p_vt=z_real_vt,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        p_dthalf=wpfloat(0.5) * p_dtime,
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
