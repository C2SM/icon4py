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

from dataclasses import dataclass

import numpy as np
from gt4py.next.common import Field
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)


@dataclass
class InterpolationState:
    """
    represents the ICON interpolation state.

    TODO [ml]: keep? does this state make sense at all?
    """

    e_bln_c_s: Field[
        [CellDim, C2EDim], float
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: Field[
        [VertexDim, V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: Field[
        [CellDim, C2EDim], float
    ]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: Field[
        [CellDim, C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: Field[
        [CellDim, C2E2CODim], float
    ]  # factor for green gauss gradient (nproma,4,nblks_c,2)
    geofac_grg_y: Field[
        [CellDim, C2E2CODim], float
    ]  # TODO combine geofac_grg_x and geofac_grg_y to tuple
    nudgecoeff_e: Field[[EdgeDim], float]  # Nudgeing coeffients for edges

    c_lin_e: Field[[EdgeDim, E2CDim], float]
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float]
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float]
    c_intp: Field[[VertexDim, V2CDim], float]
    geofac_rot: Field[[VertexDim, V2EDim], float]

    @property
    def geofac_n2s_c(self) -> Field[[CellDim], float]:
        return np_as_located_field(CellDim)(np.asarray(self.geofac_n2s)[:, 0])

    @property
    def geofac_n2s_nbh(self) -> Field[[CellDim, C2E2CDim], float]:
        return np_as_located_field(CellDim, C2E2CDim)(
            np.asarray(self.geofac_n2s)[:, 1:]
        )
