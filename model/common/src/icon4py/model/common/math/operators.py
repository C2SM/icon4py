# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2CO, C2E2CODim


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@gtx.field_operator
def _nabla2_scalar(
    psi_c: fa.CellKField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres.
    """
    nabla2_psi_c = neighbor_sum(psi_c(C2E2CO) * geofac_n2s, axis=C2E2CODim)

    return nabla2_psi_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def nabla2_scalar(
    psi_c: fa.CellKField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    nabla2_psi_c: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _nabla2_scalar(
        psi_c,
        geofac_n2s,
        out=nabla2_psi_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
