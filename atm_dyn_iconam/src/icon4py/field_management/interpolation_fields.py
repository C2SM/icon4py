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

import numpy as np
#from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

#from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.common.dimension import EdgeDim, KDim

class InterpolationFields:
    def __init__(self, run_program=True):
        self._initialized = False
        self._run_program = run_program
#        self.grid: Optional[IconGrid] = None
        self.interpolation_state = None
        self.metric_state = None
        self.vertical_params: Optional[VerticalModelParams] = None

#    def init(
#        self,
#        grid: IconGrid,
#        metric_state: MetricState,
#        interpolation_state: InterpolationState,
#        vertical_params: VerticalModelParams,
#    ):

#        self.grid = grid
#        self.metric_state: MetricState = metric_state
#        self.interpolation_state: InterpolationState = interpolation_state


    def initialization_1st_numpy(
        edge_cell_length: np.array,
        inv_dual_edge_length: np.array,
    ) -> np.array:
        c_lin_e_ = edge_cell_length[:] * inv_dual_edge_length[:]
        c_lin_e = [c_lin_e_, 1.0 - c_lin_e_]
        return c_lin_e

    def compute():
        initialization_1st_numpy(
            self.interpolation_state.edge_cell_length,
            self.interpolation_state.inv_dual_edge_length,
        )
