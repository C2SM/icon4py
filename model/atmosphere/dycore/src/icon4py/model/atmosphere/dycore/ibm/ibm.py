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
import logging

import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.state_utils import states as states_utils
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.settings import xp
from icon4py.model.common.states import prognostic_state as prog_state


"""
Immersed boundary method module

"""

log = logging.getLogger(__name__)


class ImmersedBoundaryMethod:
    """
    Main class for the immersed boundary method.
    """

    def __init__(
        self,
        icon_grid: icon_grid.IconGrid,
    ):
        self.test_value = 313

        num_cells = icon_grid.num_cells
        num_edges = icon_grid.num_edges
        num_levels = icon_grid.num_levels

        self._validate_config()

        cell_mask = xp.zeros((num_cells, num_levels), dtype=bool)
        edge_mask = xp.zeros((num_edges, num_levels), dtype=bool)

        cell_mask[313, 17] = True
        edge_mask[313, 17] = True

        self.cell_mask = gtx.as_field((CellDim, KDim), cell_mask)
        self.edge_mask = gtx.as_field((EdgeDim, KDim), edge_mask)

        log.info("IBM initialized")

    def _validate_config(self):
        log.info("IBM config validated")
        pass

    def set_boundary_conditions(
        self,
        diagnostic_state: states_utils.DiagnosticStateNonHydro,
        prognostic_state: prog_state.PrognosticState,
    ):
        log.info("IBM set BCs...")

        # cell centre variables
        prognostic_state.w = gtx.where(self.cell_mask, self.test_value, prognostic_state.w)
        prognostic_state.theta_v = gtx.where(self.cell_mask, self.test_value, prognostic_state.w)

        # edge variables
        prognostic_state.vn = gtx.where(self.edge_mask, self.test_value, prognostic_state.vn)
        log.info("IBM set BCs DONE")
