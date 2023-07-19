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
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import program
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_icon_interpolation_fields_initalization_stencil import (
    _mo_icon_interpolation_fields_initalization_stencil,
)
from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.state_utils.utils import _set_bool_c_k, _set_zero_c_k


@program(backend=gtfn_cpu.run_gtfn)
def mo_icon_interpolation_fields_initalization(
    edge_cell_length: Field[[CellDim, KDim], float],
    dual_edge_length: Field[[VertexDim, V2CDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    vert_startindex_interior_minus1: int,
    nlev: int,
):
    _mo_icon_interpolation_fields_initalization_dsl(
        edge_cell_length,
        dual_edge_length,
        out=c_lin_e,
        domain={VertexDim: (2, vert_startindex_interior_minus1), KDim: (1, nlev)},
    )
