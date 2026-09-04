# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_vp,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_to_half_levels(
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    x: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKHalfField[ta.vpfloat]:
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)
    x_ie_wp = wgtfac_e_wp * x(dims.KHalfDim + 0.5) + (wpfloat("1.0") - wgtfac_e_wp) * x(
        dims.KHalfDim - 0.5
    )
    return concat_where(
        dims.KHalfDim > 0, astype(x_ie_wp, vpfloat), astype(x(dims.KHalfDim + 0.5), vpfloat)
    )


@gtx.field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    z_kin_hor_e_wp = wpfloat("0.5") * (vn * vn + astype(vt * vt, wpfloat))
    return astype(z_kin_hor_e_wp, vpfloat)


@gtx.field_operator
def _compute_diagnostics_from_normal_wind(
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    tangential_wind = _compute_tangential_wind_vp(vn, rbf_vec_coeff_e)
    horizontal_kinetic_energy_at_edges_on_model_levels = _compute_horizontal_kinetic_energy(
        vn, tangential_wind
    )
    vn_on_half_levels = concat_where(
        dims.KHalfDim < nlev,
        _interpolate_to_half_levels(wgtfac_e, vn),
        _extrapolate_at_top(wgtfacq_e, vn),
    )

    tangential_wind_on_half_levels = (
        _interpolate_to_half_levels(wgtfac_e, tangential_wind)
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )
