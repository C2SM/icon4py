
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
DEBUG_LEVEL = 2


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

        self._dir_value_vn = 0.0
        self._dir_value_w = 0.0
        self._dir_value_theta_v = -313
        self._dir_value_p = -717

        if DEBUG_LEVEL >= 2:
            self._delta_file_vn = open("ibm_delta_vn.csv", "a")
            self._delta_file_vn.write("\n")
            self._delta_file_w = open("ibm_delta_w.csv", "a")
            self._delta_file_w.write("\n")

        log.info("IBM initialized")

    def _validate_config(self):
        log.info("IBM config validated")
        pass

    def _make_cell_mask(self, grid: icon_grid.IconGrid) -> fa.CellKField[bool]:
        cell_mask_np = np.zeros((grid.num_cells, grid.num_levels+1), dtype=bool)
        cell_mask_np[[5,16], -3:] = True
        return gtx.as_field((CellDim, KDim), cell_mask_np)
    
    def _make_edge_mask(self, grid: icon_grid.IconGrid) -> fa.EdgeKField[bool]:
        if not hasattr(self, "cell_mask"):
            raise ValueError("Cell mask must be set before edge mask.")
        cell_mask_np = self.cell_mask.ndarray
        c2e = grid.connectivities[dims.C2EDim]
        edge_mask_np = np.zeros((grid.num_edges, grid.num_levels), dtype=bool)
        for k in range(grid.num_levels):
            edge_mask_np[c2e[np.where(cell_mask_np[:,k])], k] = True
        return gtx.as_field((EdgeDim, KDim), edge_mask_np)

    def set_boundary_conditions_vn(
        self,
        vn: fa.EdgeKField[float],
    ):
        if DEBUG_LEVEL >= 1:
            vn0 = vn.ndarray.copy()

        self._set_bcs_edges(
            mask=self.edge_mask,
            dir_value=self._dir_value_vn,
            field=vn,
            out=(vn),
            offset_provider={},
        )

        if DEBUG_LEVEL >= 1:
            vn1 = vn.ndarray
            log.info(f"IBM max delta vn: {np.abs(vn1 - vn0).max()}")

    def set_boundary_conditions_w(
        self,
        theta_v_ic: fa.CellKField[float],
        z_w_expl: fa.CellKField[float],
    ):

        # Set $theta_v_{k+1/2} = 0$ as a 'hack' for setting $\gamma_{k+1/2} = 0$
        # in the tridiagonal solver. This results in:
        #  $a_{k+1/2} = 0$
        #  $b_{k+1/2} = 1$
        #  $c_{k+1/2} = 0$
        #  $d_{k+1/2} = z_w_expl_{k+1/2}$
        # and should work as theta_v_ic is not used anymore after this point.
        self._set_bcs_cells(
            mask=self.cell_mask,
            dir_value=0.,
            field=theta_v_ic,
            out=(theta_v_ic),
            offset_provider={},
        )
        # Then set the Dirichlet value for $w$.
        self._set_bcs_cells(
            mask=self.cell_mask,
            dir_value=self._dir_value_w,
            field=z_w_expl,
            out=(z_w_expl),
            offset_provider={},
        )


    def set_boundary_conditions_p(
        self,
        p: fa.CellKField[float],
    ):
        if DEBUG_LEVEL >= 1:
            p0 = p.ndarray.copy()

        self._set_bcs_cells(
            mask=self.cell_mask,
            dir_value=self._dir_value_p,
            field=p,
            out=(p),
            offset_provider={},
        )

        if DEBUG_LEVEL >= 1:
            p1 = p.ndarray
            log.info(f"IBM max delta p: {np.abs(p1 - p0 ).max()}")


    @gtx.field_operator
    def _set_bcs_cells(
        mask: fa.CellKField[bool],
        dir_value: float,
        field: fa.CellKField[float],
    ) -> fa.CellKField[float]:
        """
        Set boundary conditions for fields defined on cell centres.
        """
        field = where(mask, dir_value, field)
        return field

    @gtx.field_operator
    def _set_bcs_edges(
        mask: fa.EdgeKField[bool],
        dir_value: float,
        field: fa.EdgeKField[float],
    ) -> fa.EdgeKField[float]:
        """
        Set boundary conditions for fields defined on edges.
        """
        field = where(mask, dir_value, field)
        return field


    def check_boundary_conditions(
        self,
        prognostic_state: prognostic_state.PrognosticState,
    ):
        """
        Check boundary conditions on prognostic variables.
        """

        if DEBUG_LEVEL < 2:
            return

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