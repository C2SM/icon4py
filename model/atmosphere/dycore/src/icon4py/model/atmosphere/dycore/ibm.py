
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

        self._make_masks(grid)

        self._dirichlet_value_vn      = 0.0
        self._dirichlet_value_w       = 0.0
        self._dirichlet_value_rho     = -313.0
        self._dirichlet_value_exner   = -313.0
        self._dirichlet_value_theta_v = -313.0

        if DEBUG_LEVEL >= 2:
            self._delta_file_vn = open("ibm_delta_vn.csv", "a")
            self._delta_file_vn.write("\n")
            self._delta_file_w = open("ibm_delta_w.csv", "a")
            self._delta_file_w.write("\n")

        log.info("IBM initialized")

    def _make_masks(self, grid: icon_grid.IconGrid):
        """
        Create masks for the immersed boundary method.
        """
        half_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels+1), dtype=bool)
        half_cell_mask_np[[5,16], -3:] = True
        full_cell_mask_np = half_cell_mask_np[:, :-1]
        self.half_cell_mask = gtx.as_field((CellDim, KDim), half_cell_mask_np)
        self.full_cell_mask = gtx.as_field((CellDim, KDim), full_cell_mask_np)
    
        c2e = grid.connectivities[dims.C2EDim]
        full_edge_mask_np = np.zeros((grid.num_edges, grid.num_levels), dtype=bool)
        for k in range(grid.num_levels):
            full_edge_mask_np[c2e[np.where(full_cell_mask_np[:,k])], k] = True
        self.full_edge_mask = gtx.as_field((EdgeDim, KDim), full_edge_mask_np)

        c2e2c = grid.connectivities[dims.C2E2CDim]
        neigh_full_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels), dtype=bool)
        for k in range(grid.num_levels):
            neigh_full_cell_mask_np[c2e2c[np.where(full_cell_mask_np[:,k])], k] = True
        self.neigh_full_cell_mask = gtx.as_field((CellDim, KDim), neigh_full_cell_mask_np)

    def set_dirichlet_value_vn(
        self,
        vn: fa.EdgeKField[float],
    ):
        self.set_bcs_edges(
            mask=self.full_edge_mask,
            dir_value=self._dirichlet_value_vn,
            field=vn,
            out=vn,
            offset_provider={},
        )

    def set_dirichlet_value_w(
        self,
        w: fa.CellKField[float],
    ):
        self.set_bcs_cells(
            mask=self.half_cell_mask,
            dir_value=self._dirichlet_value_w,
            field=w,
            out=w,
            offset_provider={},
        )

    def set_dirichlet_value_rho(
        self,
        rho: fa.CellKField[float],
    ):
        self.set_bcs_cells(
            mask=self.full_cell_mask,
            dir_value=self._dirichlet_value_rho,
            field=rho,
            out=rho,
            offset_provider={},
        )

    def set_dirichlet_value_exner(
        self,
        exner: fa.CellKField[float],
    ):
        self.set_bcs_cells(
            mask=self.full_cell_mask,
            dir_value=self._dirichlet_value_exner,
            field=exner,
            out=exner,
            offset_provider={},
        )

    def set_dirichlet_value_theta_v(
        self,
        theta_v: fa.CellKField[float],
    ):
        self.set_bcs_cells(
            mask=self.full_cell_mask,
            dir_value=self._dirichlet_value_theta_v,
            field=theta_v,
            out=theta_v,
            offset_provider={},
        )

    def set_bcs_w_matrix(
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
        # and should work as theta_v_ic is not used anymore after this point,
        # nor are a, b, and c. Only alfa and beta are used in the equation for
        # exner, but those are not affected by this hack.
        self.set_bcs_cells(
            mask=self.half_cell_mask,
            dir_value=0.,
            field=theta_v_ic,
            out=theta_v_ic,
            offset_provider={},
        )
        # Then set the Dirichlet value for $w$.
        self.set_bcs_cells(
            mask=self.half_cell_mask,
            dir_value=self._dirichlet_value_w,
            field=z_w_expl,
            out=z_w_expl,
            offset_provider={},
        )

    def set_bcs_flux(
        self,
        flux: fa.EdgeKField[float],
    ):
        # Set the flux to zero at the boundaries.
        self.set_bcs_edges(
            mask=self.full_edge_mask,
            dir_value=0,
            field=flux,
            out=flux,
            offset_provider={},
        )

    def set_bcs_green_gauss_gradient(
        self,
        grad_x: fa.CellKField[float],
        grad_y: fa.CellKField[float],
    ):
        # Zero the gradients in masked cells and their neighbors.
        self.set_bcs_cells(
            mask=self.neigh_full_cell_mask,
            dir_value=0.,
            field=grad_x,
            out=grad_x,
            offset_provider={},
        )
        self.set_bcs_cells(
            mask=self.neigh_full_cell_mask,
            dir_value=0.,
            field=grad_y,
            out=grad_y,
            offset_provider={},
        )


    @gtx.field_operator
    def set_bcs_cells(
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
    def set_bcs_edges(
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

        edge_mask = self.full_edge_mask.ndarray
        cell_mask = self.half_cell_mask.ndarray
        vn = prognostic_state.vn.ndarray
        w  = prognostic_state.w.ndarray

        delta_vn = np.abs(vn[edge_mask] - self._dirichlet_value_vn)
        delta_w  = np.abs(w [cell_mask] - self._dirichlet_value_w )

        log.info(f"IBM delta on vn: min {delta_vn.min():10.3e} max {delta_vn.max():10.3e}")
        log.info(f"IBM delta on w : min {delta_w .min():10.3e} max {delta_w .max():10.3e}")

        self._delta_file_vn.write(f" {delta_vn.min():10.3e}, {delta_vn.max():10.3e},")
        self._delta_file_w .write(f" {delta_w .min():10.3e}, {delta_w .max():10.3e},")