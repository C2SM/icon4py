# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import abs, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): this stencil has no test


@field_operator
def _compute_upwind_and_antidiffusive_flux(
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    z_mflx_low = wpfloat(0.5) * (
        p_mass_flx_e * (p_cc(E2C[0]) + p_cc(E2C[1]))
        - abs(p_mass_flx_e) * (p_cc(E2C[1]) - p_cc(E2C[0]))
    )

    z_anti = p_mflx_tracer_h - z_mflx_low

    return (z_mflx_low, z_anti)


@program
def compute_upwind_and_antidiffusive_flux(
    p_mflx_tracer_h: fa.EdgeKField[wpfloat],
    p_mass_flx_e: fa.EdgeKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    z_anti: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_upwind_and_antidiffusive_flux(
        p_mflx_tracer_h,
        p_mass_flx_e,
        p_cc,
        out=(z_mflx_low, z_anti),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
