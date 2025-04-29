import logging

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.testing import serialbox as sb
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim, KDim
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.grid import (
    vertical as v_grid,
    icon as icon_grid,
)
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.states import prognostic_state

import numpy as np
import xarray as xr

"""
Immersed boundary method module

"""

log = logging.getLogger(__name__)

DO_IBM = True

class ImmersedBoundaryMethod:
    """
    Main class for the immersed boundary method.
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        savepoint_path: str,
        grid_file_path: str,
        backend: gtx.backend.Backend = gtx.gtfn_cpu,
    ):
        """
        Initialize the immersed boundary method.
        """
        self.DO_IBM = DO_IBM
        self.DEBUG_LEVEL = 2

        if not self.DO_IBM:
            return

        self._make_masks(
            grid=grid,
            savepoint_path=savepoint_path,
            grid_file_path=grid_file_path,
            backend=backend,
        )

        self._dirichlet_value_vn      = 0.0
        self._dirichlet_value_w       = 0.0
        self._dirichlet_value_rho     = 1.0
        self._dirichlet_value_exner   = 1.0
        self._dirichlet_value_theta_v = 301.0

        log.info("IBM initialized")

    def _make_masks(
        self,
        grid: icon_grid.IconGrid,
        savepoint_path: str,
        grid_file_path: str,
        backend: gtx.backend.Backend = gtx.gtfn_cpu,
    ) -> None:
        """
        Create masks for the immersed boundary method.
        """
        half_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels+1), dtype=bool)
        full_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels), dtype=bool)
        full_edge_mask_np = np.zeros((grid.num_edges, grid.num_levels), dtype=bool)
        full_vertex_mask_np = np.zeros((grid.num_vertices, grid.num_levels), dtype=bool)
        neigh_full_cell_mask_np = np.zeros((grid.num_cells, grid.num_levels), dtype=bool)

        #half_cell_mask_np = self._mask_test_cells(half_cell_mask_np)
        half_cell_mask_np = self._mask_gaussian_hill(grid, grid_file_path, savepoint_path, backend, half_cell_mask_np)

        full_cell_mask_np = half_cell_mask_np[:, :-1]

        c2e = grid.connectivities[dims.C2EDim]
        for k in range(grid.num_levels):
            full_edge_mask_np[c2e[np.where(full_cell_mask_np[:,k])], k] = True

        c2v = grid.connectivities[dims.C2VDim]
        for k in range(grid.num_levels):
            full_vertex_mask_np[c2v[np.where(full_cell_mask_np[:,k])], k] = True

        c2e2c = grid.connectivities[dims.C2E2CDim]
        for k in range(grid.num_levels):
            neigh_full_cell_mask_np[c2e2c[np.where(full_cell_mask_np[:,k])], k] = True

        self.full_cell_mask = gtx.as_field((CellDim, KDim), full_cell_mask_np)
        self.half_cell_mask = gtx.as_field((CellDim, KDim), half_cell_mask_np)
        self.full_edge_mask = gtx.as_field((EdgeDim, KDim), full_edge_mask_np)
        self.full_vertex_mask = gtx.as_field((VertexDim, KDim), full_vertex_mask_np)
        self.neigh_full_cell_mask = gtx.as_field((CellDim, KDim), neigh_full_cell_mask_np)

    def _mask_test_cells(
        self,
        half_cell_mask_np: np.ndarray
    ) -> np.ndarray:
        """
        Create a test mask.
        """
        half_cell_mask_np[[5,16], -3:] = True
        return half_cell_mask_np

    def _mask_gaussian_hill(
        self,
        grid: icon_grid.IconGrid,
        grid_file_path: str,
        savepoint_path: str,
        backend: gtx.backend.Backend,
        half_cell_mask_np: np.ndarray,
    ) -> np.ndarray:
        """
        Create a Gaussian hill mask.
        """
        hill_x = 500.
        hill_y = 500.
        hill_height = 100.
        hill_width  = 100.

        grid_file = xr.open_dataset(grid_file_path)
        data_provider = sb.IconSerialDataProvider(
            backend=backend,
            fname_prefix="icon_pydycore",
            path=savepoint_path,
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        half_level_heights = metrics_savepoint.z_ifc().asnumpy()

        compute_distance_from_hill = lambda x, y: ((x - hill_x)**2 + (y - hill_y)**2)**0.5
        compute_hill_elevation = lambda x, y: hill_height * np.exp(-(compute_distance_from_hill(x, y) / hill_width)**2)
        cell_x = grid_file.cell_circumcenter_cartesian_x.values
        cell_y = grid_file.cell_circumcenter_cartesian_y.values
        buildings = []
        buildings = [
            #[390, 400, 0, 1000,  40],
            #[497, 503, 0, 1000, 105],
            #[350, 400, 0, 1000, 75],
        ]
        for k in range(half_cell_mask_np.shape[1]):
            half_cell_mask_np[:, k] = np.where(compute_hill_elevation(cell_x, cell_y) >= half_level_heights[:,k], True, False)
            for building in buildings:
                xmin, xmax, ymin, ymax, top = building
                half_cell_mask_np[
                    (cell_x >= xmin) & (cell_x <= xmax) & (cell_y >= ymin) & (cell_y <= ymax) & (half_level_heights[:,k] <= top), k
                ] = True
        return half_cell_mask_np


    def set_dirichlet_value_vn(
        self,
        vn: fa.EdgeKField[float],
    ):
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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
        if not self.DO_IBM:
            return
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

    def set_bcs_uv_vertices(
        self,
        u_vert: fa.VertexKField[float],
        v_vert: fa.VertexKField[float],
    ):
        if not self.DO_IBM:
            return
        self.set_bcs_vertices(
            mask=self.full_vertex_mask,
            dir_value=0,
            field=u_vert,
            out=u_vert,
            offset_provider={},
        )
        self.set_bcs_vertices(
            mask=self.full_vertex_mask,
            dir_value=0,
            field=v_vert,
            out=v_vert,
            offset_provider={},
        )

    def set_bcs_khsmag(
        self,
        Kh_smag: fa.EdgeKField[float],
    ):
        if not self.DO_IBM:
            return
        # Set to zero Kh_smag as a 'hack' for setting to zero the gradient of
        # theta_v on masked edges.
        # Actually these edges have some gradient, but this is ignored for now.
        self.set_bcs_edges(
            mask=self.full_edge_mask,
            dir_value=0,
            field=Kh_smag,
            out=Kh_smag,
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

    @gtx.field_operator
    def set_bcs_vertices(
        mask: fa.VertexKField[bool],
        dir_value: float,
        field: fa.VertexKField[float],
    ) -> fa.VertexKField[float]:
        """
        Set boundary conditions for fields defined on vertices.
        """
        field = where(mask, dir_value, field)
        return field
