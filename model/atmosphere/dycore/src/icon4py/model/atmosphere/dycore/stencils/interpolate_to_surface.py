# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """
    Formerly known as _mo_solve_nonhydro_stencil_04.

    # scidoc:
    # Outputs:
    #  - interpolation_to_surface :
    #     $$
    #     \exnerprime{\ntilde}{\c}{\k-1/2} = \Wlev \exnerprime{\ntilde}{\c}{\k} + (1 - \Wlev) \exnerprime{\ntilde}{\c}{\k-1}, \quad \k \in [\max(1,\nflatlev), \nlev) \\
    #     \exnerprime{\ntilde}{\c}{\nlev-1/2} = \sum_{\k=\nlev-1}^{\nlev-3} \Wlev_{\k} \exnerprime{\ntilde}{\c}{\k}
    #     $$
    #     Interpolate the perturbation exner from full to half levels.
    #     The ground level is based on quadratic extrapolation (with
    #     hydrostatic assumption?).
    # Inputs:
    #  - $\Wlev$ : wgtfac_c
    #  - $\Wlev_{\k}$ : wgtfacq_c
    #  - $\exnerprime{\ntilde}{\c}{\k}$ : interpolant
    #
    
    """
    interpolation_to_surface = (
        wgtfacq_c(Koff[-1]) * interpolant(Koff[-1])
        + wgtfacq_c(Koff[-2]) * interpolant(Koff[-2])
        + wgtfacq_c(Koff[-3]) * interpolant(Koff[-3])
    )
    return interpolation_to_surface


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_to_surface(
    wgtfacq_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
    interpolation_to_surface: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_to_surface(
        wgtfacq_c,
        interpolant,
        out=interpolation_to_surface,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
