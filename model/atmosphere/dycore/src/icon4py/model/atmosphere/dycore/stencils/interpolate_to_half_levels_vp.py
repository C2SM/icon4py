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
def _interpolate_to_half_levels_vp(
    wgtfac_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """
    Interpolate a CellDim variable of floating precision from full levels to half levels.
    The return variable also has floating precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_c: weight factor
        interpolant: CellDim variables at full levels
    Returns:
        CellDim variables at half levels


    # scidoc:
    # Outputs:
    #  - z_exner_ic :
    #     $$
    #     \exnerprime{\ntilde}{\c}{\k-1/2} = \Wlev \exnerprime{\ntilde}{\c}{\k} + (1 - \Wlev) \exnerprime{\ntilde}{\c}{\k-1}, \quad \k \in [\max(1,\nflatlev), \nlev) \\
    #     \exnerprime{\ntilde}{\c}{\nlev-1/2} = \sum_{\k=\nlev-1}^{\nlev-3} \Wlev_{\k} \exnerprime{\ntilde}{\c}{\k}
    #     $$
    #     Interpolate the perturbation exner from full to half levels.
    #     The ground level is based on quadratic extrapolation (with
    #     hydrostatic assumption?).
    #  - z_dexner_dz_c_1 :
    #     $$
    #     \exnerprimedz{\ntilde}{\c}{\k} \approx \frac{\exnerprime{\ntilde}{\c}{\k-1/2} - \exnerprime{\ntilde}{\c}{\k+1/2}}{\Dz{\k}}, \quad \k \in [\max(1,\nflatlev), \nlev]
    #     $$
    #     Use the interpolated values to compute the vertical derivative
    #     of perturbation exner at full levels.
    #
    # Inputs:
    #  - $\Wlev$ : wgtfac_c
    #  - $\Wlev_{\k}$ : wgtfacq_c
    #  - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
    #  - $\exnerprime{\ntilde}{\c}{\k\pm1/2}$ : z_exner_ic
    #  - $1 / \Dz{\k}$ : inv_ddqz_z_full
    #


    """
    interpolation_to_half_levels_vp = wgtfac_c * interpolant + (
        vpfloat("1.0") - wgtfac_c
    ) * interpolant(Koff[-1])
    return interpolation_to_half_levels_vp


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_to_half_levels_vp(
    wgtfac_c: fa.CellKField[vpfloat],
    interpolant: fa.CellKField[vpfloat],
    interpolation_to_half_levels_vp: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_to_half_levels_vp(
        wgtfac_c,
        interpolant,
        out=interpolation_to_half_levels_vp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
