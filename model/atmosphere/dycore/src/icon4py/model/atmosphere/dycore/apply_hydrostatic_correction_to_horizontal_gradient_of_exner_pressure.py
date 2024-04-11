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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], vpfloat],
    z_hydro_corr: Field[[EdgeDim], vpfloat],
    z_gradh_exner: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_22."""
    z_gradh_exner_vp = where(ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner)
    return z_gradh_exner_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], vpfloat],
    z_hydro_corr: Field[[EdgeDim], vpfloat],
    z_gradh_exner: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        out=z_gradh_exner,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
