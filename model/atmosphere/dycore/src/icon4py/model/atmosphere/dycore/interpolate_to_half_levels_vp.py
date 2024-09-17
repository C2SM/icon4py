# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _interpolate_to_half_levels_vp(
    wgtfac_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known mo_velocity_advection_stencil_10 and as _mo_solve_nonhydro_stencil_05."""
    interpolation_to_half_levels_vp = wgtfac_c * interpolant + (
        vpfloat("1.0") - wgtfac_c
    ) * interpolant(Koff[-1])
    return interpolation_to_half_levels_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_to_half_levels_vp(
    wgtfac_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
    interpolation_to_half_levels_vp: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_to_half_levels_vp(
        wgtfac_c,
        interpolant,
        out=interpolation_to_half_levels_vp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
