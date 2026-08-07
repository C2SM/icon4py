# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid
from icon4py.model.common.math import distance_array_ns
from icon4py.model.common.states import adv_states, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.states import static_fields

log = logging.getLogger(__name__)


class TracerProfile(int, enum.Enum):
    """
    Initial tracer profile for idealized advection test cases.
    """

    #: two-dimensional smooth Gaussian curve
    GAUSSIAN_2D = 1
    #: two-dimensional smooth off-centered Gaussian curve
    GAUSSIAN_2D_OFFCENTER = 2
    #: two-dimensional discontinuous circle
    CIRCLE_2D = 3


class VelocityField(int, enum.Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: constant velocity field
    CONSTANT = 1
    #: two-dimensional divergence-free swirling velocity field
    VORTEX_2D = 2
    #: two-dimensional increasingly deformational field
    INCREASING_2D = 3


@dataclasses.dataclass
class LinearAdvectionConfig:
    tracer_profle: TracerProfile
    velocity_field: VelocityField
    cfl_number: float

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_max_velocity(
    *,
    velocity_field: VelocityField | None,
    domain_length: float,
    domain_height: float,
) -> float:
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max
    match velocity_field:
        case VelocityField.CONSTANT | VelocityField.VORTEX_2D:
            vel_max = (domain_length**2 + domain_height**2) ** 0.5
        case _:
            raise NotImplementedError(f"Velocity field {velocity_field} not implemented.")
    return vel_max


def _prepare_torus_quadratic_quadrature(
    *,
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    c2v_connectivity: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Prepare three-point quadrature rule on torus grids.

    Args:
        grid: IconGrid that entails a torus grid
        vertex_x: array that contains the vertex x-coordinates
        vertex_y: array that contains the vertex y-coordinates
        cell_center_x: array that contains the cell center x-coordinates
        cell_center_y: array that contains the cell center y-coordinates
        c2v_connectivity: array that contains the cell-to-vertex connectivity
        domain_length: length of the torus domain in x-direction
        domain_height: length of the torus domain in y-direction

    Usage:
        The return values of this function are meant to be used for setting cell averages on torus grids.
        A two-dimensional scalar function f(x,y) can be projected onto a torus plane array arr as follows:
            arr = xp.sum(weights * f(nodes[0,:,:], nodes[1,:,:]), axis=0)

    """
    array_ns = data_alloc.array_namespace(vertex_x)
    alpha = array_ns.array([[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]])
    weights_single = array_ns.array([1 / 3, 1 / 3, 1 / 3])

    n_cells = cell_center_x.shape[0]
    n_points = weights_single.size

    weights = array_ns.tile(weights_single[:, None], (1, n_cells))
    nodes = array_ns.zeros((2, n_points, n_cells))

    c2v_x = vertex_x[c2v_connectivity]
    c2v_y = vertex_y[c2v_connectivity]

    cell_to_vertex_dis_x = c2v_x - cell_center_x[:, None]
    cell_to_vertex_dis_y = c2v_y - cell_center_y[:, None]

    c2v_x = array_ns.where(
        array_ns.abs(cell_to_vertex_dis_x) > 0.5 * domain_length,
        c2v_x - array_ns.sign(cell_to_vertex_dis_x) * domain_length,
        c2v_x,
    )
    c2v_y = array_ns.where(
        array_ns.abs(cell_to_vertex_dis_y) > 0.5 * domain_height,
        c2v_y - array_ns.sign(cell_to_vertex_dis_y) * domain_height,
        c2v_y,
    )

    nodes[0, :, :] = array_ns.matmul(alpha, c2v_x.T)
    nodes[1, :, :] = array_ns.matmul(alpha, c2v_y.T)

    return weights, nodes


def _compute_idealized_velocity_field(
    *,
    velocity_field: VelocityField,
    domain_length: float,
    domain_height: float,
) -> tuple[float, float]:
    match velocity_field:
        case VelocityField.CONSTANT:
            u, v = domain_length, domain_height
        case _:
            raise NotImplementedError(f"Velocity field {velocity_field} not implemented.")
    return u, v


def _construct_idealized_prep_adv(
    *,
    velocity_field: VelocityField,
    prep_adv_state: adv_states.AdvectionPrepAdvState,
    primal_normal_x: data_alloc.NDArray,
    primal_normal_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> None:
    # we assume that the airmass is constant 1.0, the mass flux equals the velocity
    # impose 2D velocity field at time n+1/2 as required by the numerical scheme
    u, v = _compute_idealized_velocity_field(
        velocity_field,
        domain_length,
        domain_height,
    )
    vn = u * primal_normal_x + v * primal_normal_y

    vn_traj = prep_adv_state.vn_traj.ndarray
    mass_flx_me = prep_adv_state.mass_flx_me.ndarray
    mass_flx_ic = prep_adv_state.mass_flx_ic.ndarray
    vn_traj[:, :] = vn[None, :]
    mass_flx_me[:, :] = vn[None, :]
    mass_flx_ic[:, :] = 0.0


def _construct_idealized_tracer(
    *,
    tracer_profile: TracerProfile,
    tracer: data_alloc.NDArray,
    domain_center_x: float,
    domain_center_y: float,
    domain_length: float,
    domain_height: float,
    weights: data_alloc.NDArray,
    nodes: data_alloc.NDArray,
) -> None:
    array_ns = data_alloc.array_namespace(nodes)
    dx = array_ns.zeros_like(nodes[0, :, :])
    dy = array_ns.zeros_like(nodes[1, :, :])
    for i in range(nodes.shape[1]):
        dx[i, :], dy[i, :] = distance_array_ns.minimum_image_separation(
            x=nodes[0, i, :],
            y=nodes[1, i, :],
            reference_x=domain_center_x,
            reference_y=domain_center_y,
            domain_extent_x=domain_length,
            domain_extent_y=domain_height,
        )
    match tracer_profile:
        case TracerProfile.GAUSSIAN_2D:
            decay_factor = ((domain_length + domain_height) / 2.0) ** (-1.65)
            vertex_tracer = array_ns.exp(-decay_factor * (dx**2 + dy**2))
        case TracerProfile.GAUSSIAN_2D_OFFCENTER:
            dy -= domain_height / 4
            decay_factor = ((domain_length + domain_height) / 2.0) ** (-1.5)
            vertex_tracer = array_ns.exp(-decay_factor * (dx**2 + dy**2))
        case TracerProfile.CIRCLE_2D:
            radius = (domain_length + domain_height) / 8.0
            vertex_tracer = array_ns.where(dx**2 + dy**2 <= radius**2, 1.0, 0.0)
        case _:
            raise NotImplementedError(f"Initial conditions {tracer_profile} not implemented.")
    tracer[:, :] = array_ns.sum(weights * vertex_tracer, axis=0)[:, None]


def linear_advection(
    *,
    config: LinearAdvectionConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    tracer_state_now: tracer_states.TracerState,
    adv_prep_adv_state: adv_states.AdvectionPrepAdvState,
) -> None:
    """
    Initial condition for the idealized advection test case.

    """
    if tracer_state_now.qv is None:
        raise ValueError(
            "The initial condition for the linear advection test case requires the 'qv' to be active."
        )
    geometry = static_fields.geometry
    vertex_x = geometry.get(geometry_meta.VERTEX_X).ndarray
    vertex_y = geometry.get(geometry_meta.VERTEX_Y).ndarray
    cell_center_x = geometry.get(geometry_meta.CELL_CENTER_X).ndarray
    cell_center_y = geometry.get(geometry_meta.CELL_CENTER_Y).ndarray
    domain_center_x = 0.5 * grid.grid_params.domain_length
    domain_center_y = 0.5 * grid.grid_params.domain_height

    weights, nodes = _prepare_torus_quadratic_quadrature(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        c2v_connectivity=grid.connectivities[dims.C2VDim].ndarray,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )

    _construct_idealized_prep_adv(
        velocity_field=config.velocity_field,
        prep_adv_state=adv_prep_adv_state,
        primal_normal_x=geometry.get(geometry_meta.PRIMAL_NORMAL_X).ndarray,
        primal_normal_y=geometry.get(geometry_meta.PRIMAL_NORMAL_Y).ndarray,
        edge_center_x=geometry.get(geometry_meta.EDGE_CENTER_X).ndarray,
        edge_center_y=geometry.get(geometry_meta.EDGE_CENTER_Y).ndarray,
    )

    _construct_idealized_tracer(
        tracer_profile=config.tracer_profle,
        tracer=tracer_state_now.qv.ndarray,
        domain_center_x=domain_center_x,
        domain_center_y=domain_center_y,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
        weights=weights,
        nodes=nodes,
    )


def construct_reference_tracer(
    *,
    velocity_field: VelocityField,
    tracer_profile: TracerProfile,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    integration_time: float,
) -> data_alloc.NDArray:
    geometry = static_fields.geometry
    vertex_x = geometry.get(geometry_meta.VERTEX_X).ndarray
    vertex_y = geometry.get(geometry_meta.VERTEX_Y).ndarray
    cell_center_x = geometry.get(geometry_meta.CELL_CENTER_X).ndarray
    cell_center_y = geometry.get(geometry_meta.CELL_CENTER_Y).ndarray
    domain_center_x = 0.5 * grid.grid_params.domain_length
    domain_center_y = 0.5 * grid.grid_params.domain_height

    weights, nodes = _prepare_torus_quadratic_quadrature(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        c2v_connectivity=grid.connectivities[dims.C2VDim].ndarray,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )

    reference_tracer = data_alloc.zeros_like(cell_center_x)

    match velocity_field:
        case VelocityField.CONSTANT:
            # linearly shifted ICs
            u, v = _compute_idealized_velocity_field(
                velocity_field=velocity_field,
                domain_length=grid.grid_params.domain_length,
                domain_height=grid.grid_params.domain_height,
            )
            end_center_x = domain_center_x + u * integration_time
            end_center_y = domain_center_y + v * integration_time
            _construct_idealized_tracer(
                tracer_profile=tracer_profile,
                tracer=reference_tracer,
                domain_center_x=end_center_x,
                domain_center_y=end_center_y,
                domain_length=grid.grid_params.domain_length,
                domain_height=grid.grid_params.domain_height,
                weights=weights,
                nodes=nodes,
            )
        case _:
            raise NotImplementedError(
                f"Exact solution with velocity field {velocity_field} not implemented."
            )
    return reference_tracer
