# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa


@field_operator
def _compute_horizontal_tracer_flux_from_cubic_coefficients(
    p_out_e_hybrid_2: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    z_dreg_area: fa.EdgeKField[float],
) -> fa.EdgeKField[float]:
    p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

    return p_out_e_hybrid_2


@program(grid_type=GridType.UNSTRUCTURED)
def compute_horizontal_tracer_flux_from_cubic_coefficients(
    p_out_e_hybrid_2: fa.EdgeKField[float],
    p_mass_flx_e: fa.EdgeKField[float],
    z_dreg_area: fa.EdgeKField[float],
):
    _compute_horizontal_tracer_flux_from_cubic_coefficients(
        p_out_e_hybrid_2,
        p_mass_flx_e,
        z_dreg_area,
        out=p_out_e_hybrid_2,
    )
