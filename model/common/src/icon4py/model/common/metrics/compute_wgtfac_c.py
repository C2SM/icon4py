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
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_wgtfac_c_nlev(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_0(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[+1]) - z_ifc) / (z_ifc(Koff[+2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_inner(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-1]) - z_ifc(Koff[+1]))
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c(
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    wgt_fac_c = concat_where(
        (0 < dims.KDim) & (dims.KDim < nlev), _compute_wgtfac_c_inner(z_ifc), z_ifc
    )
    wgt_fac_c = concat_where(dims.KDim == 0, _compute_wgtfac_c_0(z_ifc=z_ifc), wgt_fac_c)
    wgt_fac_c = concat_where(dims.KDim == nlev, _compute_wgtfac_c_nlev(z_ifc=z_ifc), wgt_fac_c)

    return wgt_fac_c


@program(grid_type=GridType.UNSTRUCTURED, backend=run_gtfn)
def compute_wgtfac_c(
    wgtfac_c: fa.CellKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
):
    _compute_wgtfac_c(
        z_ifc,
        nlev,
        out=wgtfac_c,
    )
