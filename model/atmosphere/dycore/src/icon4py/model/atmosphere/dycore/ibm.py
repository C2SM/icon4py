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
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, Koff, VertexDim
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Immersed boundary method module

"""

log = logging.getLogger(__name__)

DIRICHLET_VALUE_VN: Final[float] = 0.0
DIRICHLET_VALUE_W: Final[float] = 0.0
DIRICHLET_VALUE_RHO: Final[float] = -999.0
DIRICHLET_VALUE_EXNER: Final[float] = -999.0
DIRICHLET_VALUE_THETA_V: Final[float] = -999.0

DIRICHLET_VALUE_FLUXES: Final[float] = 0.0
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
    # Neumann
    vn_on_half_levels = where(mask, vn(Koff[-1]), vn_on_half_levels)
    # Dirichlet
    # vn_on_half_levels = where(mask, 0.0, vn_on_half_levels)
    # Log-law?
    return vn_on_half_levels


@gtx.field_operator
def _set_bcs_green_gauss_gradient(
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


@gtx.field_operator
def _set_bcs_w_matrix(
    mask: fa.CellKField[bool],
    theta_v_at_cells_on_half_levels: fa.CellKField[float],
    w_explicit_term: fa.CellKField[float],
):
    # Set $theta_v_{k+1/2} = 0$ as a 'hack' for setting $\gamma_{k+1/2} = 0$
    # in the tridiagonal solver. This results in:
    #  $a_{k+1/2} = 0$
    #  $b_{k+1/2} = 1$
    #  $c_{k+1/2} = 0$
    #  $d_{k+1/2} = z_w_expl_{k+1/2}$
    # and works since theta_v_ic is not used anymore after this point, nor are
    # a, b, and c. Only alfa and beta are used in the equation for exner, but
    # those are not affected by this hack.
    theta_v_at_cells_on_half_levels = _set_bcs_cells(
        mask=mask,
        dir_value=0.0,
        field=theta_v_at_cells_on_half_levels,
    )
    # Then set the Dirichlet value for $w$ by modifying the right hand side.
    w_explicit_term = _set_bcs_cells(
        mask=mask,
        dir_value=0.0,
        field=w_explicit_term,
    )
    return theta_v_at_cells_on_half_levels, w_explicit_term


# ==============================================================================
# Programs

# ------------------------------------------------------------------------------
# Solve non_hydro and advection


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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_bcs_vn_gradh_w(
    mask: fa.EdgeKField[bool],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    # Set the horizontal advection of w to zero at the vertical surfaces.
    # Technically, it would be enough to set vh to zero, but vh is computed
    # from vn and vt, and vt is computed via RBF interpolation, so setting
    # the BCs on the end term is a safer approach.
    _set_bcs_edges(
        mask=mask,
        dir_value=0.0,
        field=horizontal_advection_of_w_at_edges_on_half_levels,
        out=horizontal_advection_of_w_at_edges_on_half_levels,
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
        dir_value=0.0,
        field=u_vert,
        out=u_vert,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _set_bcs_vertices(
        mask=mask,
        dir_value=0.0,
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
        mask_label: str,
        grid: icon_grid.IconGrid,
        cell_x: data_alloc.NDArray,
        cell_y: data_alloc.NDArray,
        half_level_heights: data_alloc.NDArray,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        do_ibm: bool = True,
    ):
        """
        Initialize the immersed boundary method masks.
        """

        self._make_masks(
            mask_label=mask_label,
            grid=grid,
            cell_x=cell_x,
            cell_y=cell_y,
            half_level_heights=half_level_heights,
            backend=backend,
            do_ibm=do_ibm,
        )

        log.info("IBM initialized")

    def _make_masks(
        self,
        mask_label: str,
        grid: icon_grid.IconGrid,
        cell_x: data_alloc.NDArray,
        cell_y: data_alloc.NDArray,
        half_level_heights: data_alloc.NDArray,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        do_ibm: bool,
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

        if do_ibm:
            # Fill masks, otherwise False everywhere
            log.info(f"Creating IBM masks for '{mask_label}'")
            if "gaussian_hill" in mask_label:
                half_cell_mask_np = self._mask_gaussian_hill(
                    mask_label=mask_label,
                    cell_x=cell_x,
                    cell_y=cell_y,
                    half_level_heights=half_level_heights,
                    backend=backend,
                    half_cell_mask_np=half_cell_mask_np,
                )
            elif "channel" in mask_label or "gauss3d_torus" in mask_label:
                half_cell_mask_np = self._mask_blocks(
                    mask_label=mask_label,
                    cell_x=cell_x,
                    cell_y=cell_y,
                    half_level_heights=half_level_heights,
                    backend=backend,
                    half_cell_mask_np=half_cell_mask_np,
                )
            else:
                raise ValueError(f"IBM mask_label '{mask_label}' not recognized.")

            full_cell_mask_np = half_cell_mask_np[:, :-1]

            log.info(f"Number of masked cells: {xp.sum(full_cell_mask_np)}")

            c2e = grid.connectivities[dims.C2EDim.value].ndarray
            for k in range(grid.num_levels + 1):
                half_edge_mask_np[xp.unique(c2e[xp.where(half_cell_mask_np[:, k])[0]]), k] = True
            full_edge_mask_np = half_edge_mask_np[:, :-1]

            c2v = grid.connectivities[dims.C2VDim.value].ndarray
            for k in range(grid.num_levels):
                full_vertex_mask_np[xp.unique(c2v[xp.where(full_cell_mask_np[:, k])[0]]), k] = True

            c2e2c = grid.connectivities[dims.C2E2CDim.value].ndarray
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

    def _mask_gaussian_hill(
        self,
        mask_label: str,
        cell_x: data_alloc.NDArray,
        cell_y: data_alloc.NDArray,
        half_level_heights: data_alloc.NDArray,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
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

        def compute_distance_from_hill(x, y):
            return ((x - hill_x) ** 2 + (y - hill_y) ** 2) ** 0.5

        def compute_hill_elevation(x, y):
            return hill_height * xp.exp(-((compute_distance_from_hill(x, y) / hill_width) ** 2))

        for k in range(half_cell_mask_np.shape[1]):
            half_cell_mask_np[:, k] = xp.where(
                compute_hill_elevation(cell_x, cell_y) >= half_level_heights[:, k],
                True,
                half_cell_mask_np[:, k],
            )
        return half_cell_mask_np

    def _mask_blocks(
        self,
        mask_label: str,
        cell_x: data_alloc.NDArray,
        cell_y: data_alloc.NDArray,
        half_level_heights: data_alloc.NDArray,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        half_cell_mask_np: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """
        Create a blocks mask.
        """
        xp = data_alloc.import_array_ns(backend)

        # Channel
        match mask_label:
            case "gauss3d_torus":
                # this is used for the CHANNEL_IBM testcase
                blocks = [[20000, 25500, 0, 5000, 500]]
            case "exclaim_channel_950x350x100_5m_nlev20":
                blocks = [[150, 200, 150, 199, 50]]
            case "exclaim_channel_950x350x100_2.5m_nlev40":
                blocks = [[150, 200, 149, 200, 50]]
            case "exclaim_channel_950x350x100_1.5m_nlev64":
                blocks = [[150, 200, 150, 199, 50]]
                if "multibuilding" in mask_label:
                    blocks = [
                        [150, 200, 49, 99, 50],
                        [150, 200, 150, 199, 50],
                        [150, 200, 250, 300, 50],
                        [250, 300, 49, 99, 50],
                        [250, 300, 150, 199, 50],
                        [250, 300, 250, 300, 50],
                        [350, 400, 49, 99, 50],
                        [350, 400, 150, 199, 50],
                        [350, 400, 250, 300, 50],
                        [450, 500, 49, 99, 50],
                        [450, 500, 150, 199, 50],
                        [450, 500, 250, 300, 50],
                        [550, 600, 49, 99, 50],
                        [550, 600, 150, 199, 50],
                        [550, 600, 250, 300, 50],
                    ]
            case "exclaim_channel_950x350x100_1.25m_nlev80":
                blocks = [[150, 200, 150, 199, 50]]
            case "exclaim_channel_950x350x100_1m_nlev100":
                blocks = [[150, 200, 150, 199, 50]]

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
