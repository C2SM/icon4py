# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import xarray as xr
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, Koff, VertexDim
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb


"""
Immersed boundary method module

"""

log = logging.getLogger(__name__)

DO_IBM: Final[bool] = True

DIRICHLET_VALUE_VN: Final[float] = 0.0
DIRICHLET_VALUE_W: Final[float] = 0.0
DIRICHLET_VALUE_RHO: Final[float] = -999.0
DIRICHLET_VALUE_EXNER: Final[float] = -999.0
DIRICHLET_VALUE_THETA_V: Final[float] = -999.0

DIRICHLET_VALUE_DIFFU_UV_VERT: Final[float] = 0.0

# ==============================================================================
# Field operators


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


@gtx.field_operator
def _set_bcs_vertices(
    mask: fa.VertexKField[bool],
    dir_value: float,
    field: fa.VertexKField[float],
) -> fa.VertexKField[float]:
    """
    Set boundary conditions for fields defined on vertices.
    """
    field = where(mask, dir_value, field)
    return field


@gtx.field_operator
def _set_bcs_cell_field(
    mask: fa.CellKField[bool],
    dir_field: fa.CellKField[float],
    field: fa.CellKField[float],
) -> fa.CellKField[float]:
    """
    Set boundary conditions for fields defined on cell centres.
    """
    field = where(mask, dir_field, field)
    return field


@gtx.field_operator
def _set_bcs_dvndz(
    mask: fa.EdgeKField[bool],
    vn: fa.EdgeKField[float],
    vn_on_half_levels: fa.EdgeKField[float],
) -> fa.EdgeKField[float]:
    # Set boundary conditions on d(vn)/dz by modifying the values of vn on half
    # levels.
    #
    # Neumann
    vn_on_half_levels = where(mask, vn(Koff[-1]), vn_on_half_levels)
    #
    # Dirichlet
    # vn_on_half_levels = where(mask, 0.0, vn_on_half_levels)
    #
    # Log-law?
    return vn_on_half_levels


@gtx.field_operator
def set_bcs_green_gauss_gradient(
    mask: fa.CellKField[bool],
    grad_x: fa.CellKField[float],
    grad_y: fa.CellKField[float],
):
    # Zero the gradients in masked cells (and their neighbours), used for the
    # MIURA scheme.
    grad_x = _set_bcs_cells(
        mask=mask,
        dir_value=0.0,
        field=grad_x,
    )
    grad_y = _set_bcs_cells(
        mask=mask,
        dir_value=0.0,
        field=grad_y,
    )
    return grad_x, grad_y


# ==============================================================================
# Programs

# ------------------------------------------------------------------------------
# Solve non_hydro


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_dirichlet_value_edges(
    mask: fa.EdgeKField[bool],
    dir_value: float,
    field: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _set_bcs_edges(
        mask=mask,
        dir_value=dir_value,
        field=field,
        out=field,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_dirichlet_value_cells(
    mask: fa.CellKField[bool],
    dir_value: float,
    field: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _set_bcs_cells(
        mask=mask,
        dir_value=dir_value,
        field=field,
        out=field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_bcs_dvndz(
    mask: fa.EdgeKField[bool],
    vn: fa.EdgeKField[float],
    vn_on_half_levels: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    # Set a value for vn on half levels.
    # Depending on the value, different boundary conditions can be imposed:
    #  - Neumann: vn_k+1/2 = vn_k approx for (dvn/dz)|_k+1/2 = 0
    #  - Dirichlet: vn_k+1/2 = some value
    #  - log law?
    # This only matters on the half level at the top of a masked cell(s)
    # (same as where w=0) because it is used to compute dvn/dz on the first
    # full level above that.
    _set_bcs_dvndz(
        mask=mask,
        vn=vn,
        vn_on_half_levels=vn_on_half_levels,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# ------------------------------------------------------------------------------
# Diffusion


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diffu_reset_w(
    mask: fa.CellKField[bool],
    w: fa.CellKField[float],
    w_old: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    # Restore w on masked cells
    # Setting it to zero is not what this stencil should do
    # w is zero on horizontal surfaces on top of obstacles, but it is non-zero
    # on the vertical surfaces of obstacles, and whatever value it has
    # (computed by solve non-hydro) should be maintained. Horizontal diffusion
    # would diffuse it with unphysical w values coming from inside the
    # obstacles, so here it is reset.
    _set_bcs_cell_field(
        mask=mask,
        dir_field=w_old,
        field=w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diffu_set_bcs_uv_vertices(
    mask: fa.VertexKField[bool],
    u_vert: fa.VertexKField[float],
    v_vert: fa.VertexKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    # Set to zero the horizontal wind on masked vertices after it is computed
    # in diffusion, to be used for computing nabla2(hori_wind).
    _set_bcs_vertices(
        mask=mask,
        dir_value=DIRICHLET_VALUE_DIFFU_UV_VERT,
        field=u_vert,
        out=u_vert,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _set_bcs_vertices(
        mask=mask,
        dir_value=DIRICHLET_VALUE_DIFFU_UV_VERT,
        field=v_vert,
        out=v_vert,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


class ImmersedBoundaryMethodMasks:
    """
    Container for the immersed boundary method masks.
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        savepoint_path: str,
        grid_file_path: str,
        backend: gtx_typing.Backend,
    ):
        """
        Initialize the immersed boundary method masks.
        """

        self._make_masks(
            grid=grid,
            savepoint_path=savepoint_path,
            grid_file_path=grid_file_path,
            backend=backend,
        )

        log.info("IBM initialized")

    def _make_masks(
        self,
        grid: icon_grid.IconGrid,
        savepoint_path: str,
        grid_file_path: str,
        backend: gtx_typing.Backend,
    ) -> None:
        """
        Create masks for the immersed boundary method.
        """
        xp = data_alloc.import_array_ns(backend)

        half_cell_mask_np = xp.zeros((grid.num_cells, grid.num_levels + 1), dtype=bool)
        half_edge_mask_np = xp.zeros((grid.num_edges, grid.num_levels + 1), dtype=bool)
        full_cell_mask_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=bool)
        full_edge_mask_np = xp.zeros((grid.num_edges, grid.num_levels), dtype=bool)
        full_vertex_mask_np = xp.zeros((grid.num_vertices, grid.num_levels), dtype=bool)
        neigh_full_cell_mask_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=bool)

        if DO_IBM:
            # Fill masks, otherwise False everywhere
            # half_cell_mask_np = self._mask_test_cells(half_cell_mask_np)
            # half_cell_mask_np = self._mask_gaussian_hill(grid_file_path, savepoint_path, backend, half_cell_mask_np)
            half_cell_mask_np = self._mask_blocks(
                grid_file_path, savepoint_path, backend, half_cell_mask_np
            )

            full_cell_mask_np = half_cell_mask_np[:, :-1]

            log.info(f"IBM: nr. of masked cells: {xp.sum(full_cell_mask_np)}")

            c2e = grid.connectivities[dims.C2EDim]
            for k in range(grid.num_levels + 1):
                half_edge_mask_np[xp.unique(c2e[xp.where(half_cell_mask_np[:, k])[0]]), k] = True
            full_edge_mask_np = half_edge_mask_np[:, :-1]

            c2v = grid.connectivities[dims.C2VDim]
            for k in range(grid.num_levels):
                full_vertex_mask_np[xp.unique(c2v[xp.where(full_cell_mask_np[:, k])[0]]), k] = True

            c2e2c = grid.connectivities[dims.C2E2CDim]
            for k in range(grid.num_levels):
                neigh_full_cell_mask_np[
                    xp.unique(c2e2c[xp.where(full_cell_mask_np[:, k])[0]]), k
                ] = True

        self.full_cell_mask = gtx.as_field((CellDim, KDim), full_cell_mask_np)
        self.half_cell_mask = gtx.as_field((CellDim, KDim), half_cell_mask_np)
        self.full_edge_mask = gtx.as_field((EdgeDim, KDim), full_edge_mask_np)
        self.half_edge_mask = gtx.as_field((EdgeDim, KDim), half_edge_mask_np)
        self.full_vertex_mask = gtx.as_field((VertexDim, KDim), full_vertex_mask_np)
        self.neigh_full_cell_mask = gtx.as_field((CellDim, KDim), neigh_full_cell_mask_np)

    def _mask_test_cells(self, half_cell_mask_np: data_alloc.NDArray) -> data_alloc.NDArray:
        """
        Create a test mask.
        """

        # on Torus_Triangles_1000m_x_1000m_res10m
        #
        # individual cells
        # half_cell_mask_np[[10,11], -3:] = True
        #
        # "cube" block (x-aligned faces zig-zaggy)
        # ids = [
        #     1449, 1450, 1451,
        #     1597, 1598, 1599, 1600, 1601, 1602, 1603,
        #     1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755,
        #     1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907,
        #     2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059,
        #     2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211,
        #     2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362,
        #     2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510,
        #     2652, 2653, 2654, 2655, 2656, 2657, 2658,
        #     2804, 2805, 2806
        # ]
        #
        # non-zig-zaggy block
        # ids = [
        #     1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451,
        #     1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603,
        #     1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755,
        #     1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907,
        #     2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059,
        #     2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211,
        #     2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362,
        #     2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512,
        #     2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662,
        #     2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812
        # ]
        # half_cell_mask_np[ids, -81:] = True

        # on Torus_Triangles_250m_x_250m_res2.5m

        return half_cell_mask_np

    def _mask_gaussian_hill(
        self,
        grid_file_path: str,
        savepoint_path: str,
        backend: gtx_typing.Backend,
        half_cell_mask_np: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """
        Create a Gaussian hill mask.
        """
        xp = data_alloc.import_array_ns(backend)

        hill_x = 500.0
        hill_y = 500.0
        hill_height = 100.0
        hill_width = 100.0

        grid_file = xr.open_dataset(grid_file_path)
        data_provider = sb.IconSerialDataProvider(
            backend=backend,
            fname_prefix="icon_pydycore",
            path=savepoint_path,
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        half_level_heights = metrics_savepoint.z_ifc().ndarray

        compute_distance_from_hill = lambda x, y: ((x - hill_x) ** 2 + (y - hill_y) ** 2) ** 0.5
        compute_hill_elevation = lambda x, y: hill_height * xp.exp(
            -((compute_distance_from_hill(x, y) / hill_width) ** 2)
        )
        cell_x = xp.asarray(grid_file.cell_circumcenter_cartesian_x.values)
        cell_y = xp.asarray(grid_file.cell_circumcenter_cartesian_y.values)
        for k in range(half_cell_mask_np.shape[1]):
            half_cell_mask_np[:, k] = xp.where(
                compute_hill_elevation(cell_x, cell_y) >= half_level_heights[:, k],
                True,
                half_cell_mask_np[:, k],
            )
        return half_cell_mask_np

    def _mask_blocks(
        self,
        grid_file_path: str,
        savepoint_path: str,
        backend: gtx_typing.Backend,
        half_cell_mask_np: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """
        Create a blocks mask.
        """
        xp = data_alloc.import_array_ns(backend)

        # on Torus_Triangles_1000m_x_1000m_res10m
        # blocks = [
        #     #
        #     [450, 550, 450, 550, 100], # 1x1 100x100x100
        #     #[  0, 1000,  0, 1000, 1.5], # flat plane
        #     #
        #     #[200, 300, 200, 300, 100], # 2x2 100x100x100
        #     #[700, 800, 200, 300, 100], # 2x2 100x100x100
        #     #[200, 300, 695, 800, 100], # 2x2 100x100x100
        #     #[700, 800, 695, 800, 100], # 2x2 100x100x100
        #     #
        #     #[ 75, 175,  75, 180, 100], # 4x4 100x100x100
        #     #[325, 425,  75, 180, 100], # 4x4 100x100x100
        #     #[575, 675,  75, 180, 100], # 4x4 100x100x100
        #     #[825, 925,  75, 180, 100], # 4x4 100x100x100
        #     #[ 75, 175, 325, 430, 100], # 4x4 100x100x100
        #     #[325, 425, 325, 430, 100], # 4x4 100x100x100
        #     #[575, 675, 325, 430, 100], # 4x4 100x100x100
        #     #[825, 925, 325, 430, 100], # 4x4 100x100x100
        #     #[ 75, 175, 575, 680, 100], # 4x4 100x100x100
        #     #[325, 425, 575, 680, 100], # 4x4 100x100x100
        #     #[575, 675, 575, 680, 100], # 4x4 100x100x100
        #     #[825, 925, 575, 680, 100], # 4x4 100x100x100
        #     #[ 75, 175, 825, 930, 100], # 4x4 100x100x100
        #     #[325, 425, 825, 930, 100], # 4x4 100x100x100
        #     #[575, 675, 825, 930, 100], # 4x4 100x100x100
        #     #[825, 925, 825, 930, 100], # 4x4 100x100x100
        #     #
        #     #[400, 420, 485, 510,  80], # on hill front
        #     #[580, 600, 485, 510,  80], # on hill back
        #     #[495, 520, 395, 420,  80], # on hill right
        #     #[495, 520, 580, 600,  80], # on hill left
        #     #[490, 510, 485, 510, 120], # on hill top
        # ]

        # # on Torus_Triangles_250m_x_250m_res2.5m / res1.25m
        # blocks = [
        #     #
        #     [75, 175, 78, 178, 100],  # 1x1 100x100x100
        # ]

        # Channel
        match savepoint_path.split("/")[-2]:
            case "exclaim_channel_950x350x100_5m_nlev20":
                blocks = [[150, 200, 150, 199, 50]]
            case "exclaim_channel_950x350x100_2.5m_nlev40":
                blocks = [[150, 200, 149, 200, 50]]
            case "exclaim_channel_950x350x100_1.5m_nlev64":
                blocks = [[150, 200, 150, 199, 50]]
            case "exclaim_channel_950x350x100_1.25m_nlev80":
                blocks = [[150, 200, 150, 199, 50]]
            case "exclaim_channel_950x350x100_1m_nlev100":
                blocks = [[150, 200, 150, 199, 50]]

        grid_file = xr.open_dataset(grid_file_path)
        data_provider = sb.IconSerialDataProvider(
            backend=backend,
            fname_prefix="icon_pydycore",
            path=savepoint_path,
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        half_level_heights = metrics_savepoint.z_ifc().ndarray

        cell_x = xp.asarray(grid_file.cell_circumcenter_cartesian_x.values)
        cell_y = xp.asarray(grid_file.cell_circumcenter_cartesian_y.values)
        for k in range(half_cell_mask_np.shape[1]):
            for block in blocks:
                xmin, xmax, ymin, ymax, top = block
                half_cell_mask_np[:, k] = xp.where(
                    (cell_x >= xmin)
                    & (cell_x <= xmax)
                    & (cell_y >= ymin)
                    & (cell_y <= ymax)
                    & (half_level_heights[:, k] <= top),
                    True,
                    half_cell_mask_np[:, k],
                )
        return half_cell_mask_np

    # --------------------------------------------------------------------------
    # non-hydro and advection part

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
        _set_bcs_cells(
            mask=self.half_cell_mask,
            dir_value=0.0,
            field=theta_v_ic,
            out=theta_v_ic,
            offset_provider={},
        )
        # Then set the Dirichlet value for $w$.
        _set_bcs_cells(
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
        _set_bcs_edges(
            mask=self.full_edge_mask,
            dir_value=0,
            field=flux,
            out=flux,
            offset_provider={},
        )

    def set_bcs_gradh_w(
        self,
        horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[float],
    ):
        if not self.DO_IBM:
            return
        # Set the horizontal advection of w to zero at the vertical surfaces
        _set_bcs_edges(
            mask=self.half_edge_mask,
            dir_value=0,
            field=horizontal_advection_of_w_at_edges_on_half_levels,
            out=horizontal_advection_of_w_at_edges_on_half_levels,
            offset_provider={},
        )

    # --------------------------------------------------------------------------
    # diffusion part

    def set_bcs_khsmag(
        self,
        Kh_smag: fa.EdgeKField[float],
    ):
        if not self.DO_IBM:
            return
        # Set to zero Kh_smag as a 'hack' for setting to zero the gradient of
        # theta_v on masked edges.
        # Actually these edges have some gradient, but this is ignored for now.
        _set_bcs_edges(
            mask=self.full_edge_mask,
            dir_value=0,
            field=Kh_smag,
            out=Kh_smag,
            offset_provider={},
        )
