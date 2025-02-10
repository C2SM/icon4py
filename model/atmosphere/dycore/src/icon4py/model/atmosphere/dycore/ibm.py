
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
DEBUG = True


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

        self._dir_value_w = 0.0
        self._dir_value_theta_v = -313
        self._dir_value_vn = 0.0

        if DEBUG:
            self._delta_file_vn = open("ibm_delta_vn.csv", "a")
            self._delta_file_vn.write("\n")
            self._delta_file_w = open("ibm_delta_w.csv", "a")
            self._delta_file_w.write("\n")

        log.info("IBM initialized")

    def _validate_config(self):
        log.info("IBM config validated")
        pass

    def _make_cell_mask(self, grid: icon_grid.IconGrid) -> fa.CellKField[bool]:
        cell_mask = np.zeros((grid.num_cells, grid.num_levels+1), dtype=bool)
        cell_mask[[5,16], -2:] = True
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

        if DEBUG:
            vn0 = prognostic_state.vn.ndarray.copy()
            w0  = prognostic_state.w.ndarray.copy()

        self._set_bcs_cells(
            mask=self.cell_mask,
            dir_value_w=self._dir_value_w,
            dir_value_theta_v=self._dir_value_theta_v,
            w=prognostic_state.w,
            theta_v=prognostic_state.theta_v,
            out=(prognostic_state.w, prognostic_state.theta_v),
            offset_provider={},
        )
        self._set_bcs_edges(
            mask=self.edge_mask,
            dir_value_vn=self._dir_value_vn,
            vn=prognostic_state.vn,
            out=(prognostic_state.vn),
            offset_provider={},
        )

        if DEBUG:
            vn1 = prognostic_state.vn.ndarray
            w1  = prognostic_state.w.ndarray
            log.info(f"IBM max delta vn: {np.abs(vn1 - vn0).max()}")
            log.info(f"IBM max delta w : {np.abs(w1  - w0 ).max()}")

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

    def check_boundary_conditions(
        self,
        prognostic_state: prognostic_state.PrognosticState,
    ):
        """
        Check boundary conditions on prognostic variables.
        """
        edge_mask = self.edge_mask.ndarray
        cell_mask = self.cell_mask.ndarray
        vn = prognostic_state.vn.ndarray
        w  = prognostic_state.w.ndarray

        delta_vn = np.abs(vn[edge_mask] - self._dir_value_vn)
        delta_w  = np.abs(w [cell_mask] - self._dir_value_w )

        log.info(f"IBM delta on vn: min {delta_vn.min():10.3e} max {delta_vn.max():10.3e}")
        log.info(f"IBM delta on w : min {delta_w .min():10.3e} max {delta_w .max():10.3e}")

        self._delta_file_vn.write(f" {delta_vn.min():10.3e}, {delta_vn.max():10.3e},")
        self._delta_file_w .write(f" {delta_w .min():10.3e}, {delta_w .max():10.3e},")