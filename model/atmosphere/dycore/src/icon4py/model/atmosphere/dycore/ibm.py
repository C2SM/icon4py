
import logging

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.states import prognostic_state

import numpy as np

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
        grid: icon_grid.IconGrid,
    ):
        """
        Initialize the immersed boundary method.
        """
        self._validate_config()

        self.cell_mask = self._make_cell_mask(grid)
        self.edge_mask = self._make_edge_mask(grid)

        log.info("IBM initialized")

    def _validate_config(self):
        log.info("IBM config validated")
        pass

    def _make_cell_mask(self, grid: icon_grid.IconGrid) -> fa.CellKField[bool]:
        cell_mask = np.zeros((grid.num_cells, grid.num_levels), dtype=bool)
        cell_mask[2:4, -2:] = True
        return gtx.as_field((CellDim, KDim), cell_mask)
    
    def _make_edge_mask(self, grid: icon_grid.IconGrid) -> fa.EdgeKField[bool]:
        if not hasattr(self, "cell_mask"):
            raise ValueError("Cell mask must be set before edge mask.")
        cell_mask = self.cell_mask.ndarray
        c2e = grid.connectivities[dims.C2EDim]
        edge_mask = np.zeros((grid.num_edges, grid.num_levels), dtype=bool)
        for k in range(grid.num_levels):
            edge_mask[c2e[np.where(cell_mask[:,k])], k] = True
        return gtx.as_field((EdgeDim, KDim), edge_mask)

    def set_boundary_conditions(
        self,
        prognostic_state: prognostic_state.PrognosticState,
    ):
        """
        Set boundary conditions on prognostic variables.
        """
        log.info("IBM set BCs...")

        self._set_bcs_cells(
            mask=self.cell_mask,
            dir_value_w=0.0,
            dir_value_theta_v=-313.0,
            w=prognostic_state.w,
            theta_v=prognostic_state.theta_v,
            out=(prognostic_state.w, prognostic_state.theta_v),
            offset_provider={},
        )
        self._set_bcs_edges(
            mask=self.edge_mask,
            dir_value_vn=0.0,
            vn=prognostic_state.vn,
            out=(prognostic_state.vn),
            offset_provider={},
        )

    @gtx.field_operator
    def _set_bcs_cells(
        mask: fa.CellKField[bool],
        dir_value_w: float,
        dir_value_theta_v: float,
        w: fa.CellKField[float],
        theta_v: fa.CellKField[float],
    ) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
        """
        Set boundary conditions for variables defined on cell centres.
        """
        w       = where(mask, dir_value_w,       w)
        #theta_v = where(mask, dir_value_theta_v, theta_v)
        return w, theta_v

    @gtx.field_operator
    def _set_bcs_edges(
        mask: fa.EdgeKField[bool],
        dir_value_vn: float,
        vn: fa.EdgeKField[float],
    ) -> fa.EdgeKField[float]:
        """
        Set boundary conditions for variables defined on edges.
        """
        vn = where(mask, dir_value_vn, vn)
        return vn
