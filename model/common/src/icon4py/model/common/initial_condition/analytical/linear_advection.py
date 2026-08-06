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
import numpy as np
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    icon as icon_grid,
)
from icon4py.model.common.math import distance_array_ns
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.atmosphere.tracer_advection import tracer_advection_states
from icon4py.model.common.states import tracer_states



if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
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
    #: initial tracer profile
    tracer_profle: TracerProfile
    #: velocity field
    velocity_field: VelocityField
    #: cfl number
    cfl_number: float

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_max_velocity(
    velocity_field: VelocityField,
    domain_length: float,
    domain_height: float,
) -> float:
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max
    match velocity_field:
        case VelocityField.CONSTANT | VelocityField.VORTEX_2D:
            vel_max = (domain_length**2 + domain_height**2) ** 0.5
        case _:
            raise NotImplementedError(
                f"Velocity field {velocity_field} not implemented."
            )
    return vel_max


def _get_torus_dimensions(domain_size: float):
    return 0.5 * domain_size, 0.5 * domain_size, domain_size, domain_size


def _prepare_torus_quadratic_quadrature(
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    c2v_connectivity: data_alloc.NDArray,
    min_edge_length: float,
):
    """
    Prepare three-point quadrature rule on torus grids.

    Args:
        grid: input argument, IconGrid that entails a torus grid
        vertex_x: input argument, array that contains the vertex x-coordinates
        vertex_y: input argument, array that contains the vertex y-coordinates
        cell_center_x: input argument, array that contains the cell center x-coordinates
        cell_center_y: input argument, array that contains the cell center y-coordinates
        c2v_connectivity: input argument, array that contains the cell-to-vertex connectivity
        min_edge_length: input argument, the smallest edge length in the grid

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

    nodes[0, :, :] = array_ns.matmul(alpha, c2v_x.T)
    nodes[1, :, :] = array_ns.matmul(alpha, c2v_y.T)

    # revert to cell centers for degenerate triangles at the domain boundary due to periodicity
    node_x_diff = c2v_x - array_ns.roll(c2v_x, 1, axis=1)
    node_y_diff = c2v_y - array_ns.roll(c2v_y, 1, axis=1)
    node_dist_max = array_ns.max(array_ns.sqrt(node_x_diff**2 + node_y_diff**2), axis=1)
    mask = node_dist_max > 2.0 * min_edge_length
    weights[:, mask] = 1 / n_points
    nodes[:, :, mask] = array_ns.stack((cell_center_x[None, mask], cell_center_y[None, mask]))

    return weights, nodes


def _compute_idealized_velocity_field(
    velocity_field: VelocityField,
    domain_length: float,
    domain_height: float,
):
    match velocity_field:
        case VelocityField.CONSTANT:
            u, v = domain_length, domain_height
        case _:
            raise NotImplementedError(
                f"Velocity field {velocity_field} not implemented."
            )
    return u, v


def _construct_idealized_prep_adv(
    velocity_field: VelocityField,
    prep_adv_state: tracer_advection_states.AdvectionPrepAdvState,
    primal_normal_x: data_alloc.NDArray,
    primal_normal_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> tracer_advection_states.AdvectionPrepAdvState:
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
    vn_traj[:,:] = vn[None, :]
    mass_flx_me[:,:] = vn[None, :]
    mass_flx_ic[:,:] = 0.0


def _construct_idealized_tracer(
    tracer_profile: TracerProfile,
    tracer: data_alloc.NDArray,
    domain_center_x: float,
    domain_center_y: float,
    domain_length: float,
    domain_height: float,
    weights: data_alloc.NDArray,
    nodes: data_alloc.NDArray,
):
    array_ns = data_alloc.array_namespace(nodes)
    # impose tracer IC at the horizontal grid center
    dx, dy = distance_array_ns.minimum_image_separation(
        x=nodes[0, :, :],
        y=nodes[1, :, :],
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
            raise NotImplementedError(
                f"Initial conditions {test_config.initial_conditions} not implemented."
            )
    tracer[:,:] = array_ns.sum(weights * vertex_tracer, axis=0)[:, array_ns.newaxis]


def linear_advection(  # noqa: PLR0915 [too-many-statements]
    *,
    config: LinearAdvectionConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    tracer_state_now: tracer_states.TracerState,
    adv_prep_adv_state: tracer_advection_states.AdvectionPrepAdvState,
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
    min_edge_length = geometry.get(geometry_meta.EDGE_LENGTH).ndarray.min()
    (
        domain_center_x,
        domain_center_y,
        domain_length,
        domain_height,
    ) = _get_torus_dimensions(grid.grid_params.domain_length)

    weights, nodes = _prepare_torus_quadratic_quadrature(
        grid, vertex_x, vertex_y, cell_center_x, cell_center_y, min_edge_length
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
        domain_length=domain_length,
        domain_height=domain_height,
        weights=weights,
        nodes=nodes,
    )


def construct_reference_tracer_numpy(
    test_config,
    icon_grid,
    x_center,
    y_center,
    x_range,
    y_range,
    edges_center_x,
    edges_center_y,
    node_x,
    node_y,
    integration_time,
    weights,
    nodes,
) -> data_alloc.NDArray:
    match test_config.velocity_field:
        case VelocityField.CONSTANT:
            # linearly shifted ICs
            u, v = _compute_idealized_velocity_field(
                test_config, x_range, y_range, edges_center_x, edges_center_y, time, time_end
            )
            x = nodes[0, :, :] - (x_center + u * time)
            y = nodes[1, :, :] - (y_center + v * time)
            tracer = array_ns.sum(
                weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
            )
        case VelocityField.VORTEX_2D:
            # ICs
            x = nodes[0, :, :] - x_center
            y = nodes[1, :, :] - y_center
            tracer = array_ns.sum(
                weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
            )
        case VelocityField.INCREASING_2D:
            # shifted and deformed ICs
            et = array_ns.exp(time)
            emt = array_ns.exp(-time)
            x = -emt * (-x_range + x_range * et - x_range * time - nodes[0, :, :]) - x_center
            y = -y_range + y_range * et - y_range * et * time + nodes[1, :, :] * et - y_center
            tracer = array_ns.sum(
                weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
            )
        case _:
            raise NotImplementedError(
                f"Exact solution with velocity field {test_config.velocity_field} not implemented."
            )