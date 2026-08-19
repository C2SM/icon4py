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
import math
import typing
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common.config import config_io, options as common_conf_opt
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid
from icon4py.model.common.math import distance_array_ns
from icon4py.model.common.metrics import metrics_attributes as metrics_meta
from icon4py.model.common.states import (
    prep_adv_states,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.states import static_fields


@config_io.register_enum
class HorizontalTracerProfile(int, enum.Enum):
    """
    Initial tracer profile for idealized advection test cases.
    """

    #: two-dimensional smooth Gaussian curve
    GAUSSIAN_2D = 1
    #: two-dimensional smooth off-centered Gaussian curve
    GAUSSIAN_2D_OFFCENTER = 2
    #: one-dimensional smooth Gaussian curve in x direction
    GAUSSIAN_1D_X = 3
    #: one-dimensional smooth Gaussian curve in y direction
    GAUSSIAN_1D_Y = 4
    #: two-dimensional discontinuous circle
    CIRCLE_2D = 5


@config_io.register_enum
class HorizontalVelocityField(int, enum.Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: uniform velocity field in x and y directions
    UNIFORM_XY = 1
    #: uniform velocity field in x direction, zero velocity in y direction
    UNIFORM_X = 2
    #: uniform velocity field in y direction, zero velocity in x direction
    UNIFORM_Y = 3
    #: two-dimensional divergence-free swirling velocity field
    VORTEX_2D = 4
    #: two-dimensional increasingly deformational field
    INCREASING_2D = 5


@dataclasses.dataclass
class LinearHorizontalAdvectionConfig:
    tracer_profile: typing.Annotated[
        HorizontalTracerProfile,
        common_conf_opt.ConfigOption(
            description="Initial tracer profile.",
            icon_equivalent=None,
        ),
    ] = HorizontalTracerProfile.GAUSSIAN_2D
    velocity_field: typing.Annotated[
        HorizontalVelocityField,
        common_conf_opt.ConfigOption(
            description="Velocity field for transporting the tracer.",
            icon_equivalent=None,
        ),
    ] = HorizontalVelocityField.UNIFORM_XY
    cfl_number: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum CFL number for determination of the time step.",
            icon_equivalent=None,
        ),
    ] = 0.8
    initial_center: typing.Annotated[
        tuple[float, float],
        common_conf_opt.ConfigOption(
            description="Initial center of the tracer profile relative to the model domain size in x and y directions.",
            icon_equivalent=None,
        ),
    ] = (0.5, 0.5)
    decay_radius: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Decay radius for the Gaussian tracer profile (0.001 fraction), relative to the model domain size.",
            icon_equivalent=None,
        ),
    ] = 0.25

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_max_velocity(
    *,
    velocity_field: HorizontalVelocityField,
    domain_length: float,
    domain_height: float,
) -> float:
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max
    u, v = _compute_idealized_horizontal_velocity_field(
        velocity_field=velocity_field,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    return (u**2 + v**2) ** 0.5


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
    Prepare three-point second-order-accuracy quadrature rule on torus grids.
    Triangular cells must be uniform to guarantee second-order accuracy.
    Args:
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


def _compute_tracer_center(
    *,
    initial_center: tuple[float, float],
    origin_x: float,
    origin_y: float,
    domain_length: float,
    domain_height: float,
    displacement_x: float = 0.0,
    displacement_y: float = 0.0,
) -> tuple[float, float]:
    """
    Center of the tracer profile after being displaced, wrapped back into the torus domain.

    ``initial_center`` is given as a fraction of the domain extent, relative to the domain
    origin ``(origin_x, origin_y)``, which is not necessarily at zero.
    """
    return (
        origin_x + (initial_center[0] * domain_length + displacement_x) % domain_length,
        origin_y + (initial_center[1] * domain_height + displacement_y) % domain_height,
    )


def _compute_idealized_horizontal_velocity_field(
    *,
    velocity_field: HorizontalVelocityField,
    domain_length: float,
    domain_height: float,
) -> tuple[float, float]:
    """
    Create an idealized velocity field for the linear horizontal advection test case.

    Args:
        velocity_field: velocity field type
        domain_length: domain size in x-direction
        domain_height: domain size in y-direction

    Returns:
        tuple[float, float]: velocity components (u, v) in x and y directions
    """
    match velocity_field:
        case HorizontalVelocityField.UNIFORM_XY:
            u, v = domain_length, domain_height
        case HorizontalVelocityField.UNIFORM_X:
            u, v = domain_length, 0.0
        case HorizontalVelocityField.UNIFORM_Y:
            u, v = 0.0, domain_height
        case _:
            raise NotImplementedError(f"Velocity field {velocity_field} not implemented.")
    return u, v


def _fill_prep_adv_from_prescribed_wind_field(
    *,
    velocity_field: HorizontalVelocityField,
    prep_adv_state: prep_adv_states.TracerPrepAdvState,
    primal_normal_x: data_alloc.NDArray,
    primal_normal_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> None:
    # we assume that the airmass is a constant value 1.0, the mass flux equals the velocity
    # impose 2D velocity field at time n+1/2 as required by the numerical scheme
    u, v = _compute_idealized_horizontal_velocity_field(
        velocity_field=velocity_field,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    vn = u * primal_normal_x + v * primal_normal_y

    vn_traj = prep_adv_state.vn_traj.ndarray
    mass_flx_me = prep_adv_state.mass_flx_me.ndarray
    mass_flx_ic = prep_adv_state.mass_flx_ic.ndarray
    vn_traj[:, :] = vn[:, None]
    mass_flx_me[:, :] = vn[:, None]
    mass_flx_ic[:, :] = 0.0


def _fill_tracer_from_analytical_profile(
    *,
    config: LinearHorizontalAdvectionConfig,
    tracer_buffer: data_alloc.NDArray,
    weights: data_alloc.NDArray,
    nodes: data_alloc.NDArray,
    tracer_center_x: float,
    tracer_center_y: float,
    domain_length: float,
    domain_height: float,
) -> None:
    """
    Create an idealized tracer profile on a torus grid, given the center of the tracer and
    the domain size. When the tracer profile is Gaussian, the decay factor is chosen such
    that the Gaussian decays to 1.e-3 at the decay radius from the center.
    Otherwise, the tracer profile is a discontinuous circle with radius equal to 1/4 of
    the average domain size. The accuracy of the tracer profile depends on the number of
    quadrature points (nodes) used to project the profile onto the grid.

    Args:
        config: configuration for the linear horizontal advection test case
        tracer_buffer: buffer array to store the tracer values
        tracer_center_x: tracer center x-coordinate
        tracer_center_y: tracer center y-coordinate
        domain_length: domain size in x-direction
        domain_height: domain size in y-direction
        weights: quadrature weights
        nodes: quadrature nodes
    """
    array_ns = data_alloc.array_namespace(nodes)
    dx = array_ns.zeros_like(nodes[0, :, :])
    dy = array_ns.zeros_like(nodes[1, :, :])
    for i in range(nodes.shape[1]):
        dx[i, :], dy[i, :] = distance_array_ns.minimum_image_separation(
            x=nodes[0, i, :],
            y=nodes[1, i, :],
            reference_x=tracer_center_x,
            reference_y=tracer_center_y,
            domain_extent_x=domain_length,
            domain_extent_y=domain_height,
        )
    match config.tracer_profile:
        case HorizontalTracerProfile.GAUSSIAN_2D:
            decay_factor = (
                -1.0
                / (config.decay_radius * min(domain_length, domain_height)) ** (2)
                * math.log(1e-3)
            )
            vertex_tracer = array_ns.exp(-decay_factor * (dx**2 + dy**2))
        case HorizontalTracerProfile.GAUSSIAN_2D_OFFCENTER:
            decay_factor = (
                -1.0
                / (config.decay_radius * min(domain_length, domain_height)) ** (2)
                * math.log(1e-3)
            )
            dy -= domain_height / 4
            vertex_tracer = array_ns.exp(-decay_factor * (dx**2 + dy**2))
        case HorizontalTracerProfile.CIRCLE_2D:
            radius = (domain_length + domain_height) / 8.0
            vertex_tracer = array_ns.where(dx**2 + dy**2 <= radius**2, 1.0, 0.0)
        case HorizontalTracerProfile.GAUSSIAN_1D_X:
            decay_factor = -1.0 / (config.decay_radius * domain_length) ** (2) * math.log(1e-3)
            vertex_tracer = array_ns.exp(-decay_factor * (dx**2))
        case HorizontalTracerProfile.GAUSSIAN_1D_Y:
            decay_factor = -1.0 / (config.decay_radius * domain_height) ** (2) * math.log(1e-3)
            vertex_tracer = array_ns.exp(-decay_factor * (dy**2))
        case _:
            raise NotImplementedError(
                f"Initial tracer profile {config.tracer_profile} not implemented."
            )
    tracer_buffer[:, :] = array_ns.sum(weights * vertex_tracer, axis=0)[:, None]


def linear_horizontal_advection(
    *,
    config: LinearHorizontalAdvectionConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    tracer_prep_adv_state: prep_adv_states.TracerPrepAdvState,
) -> None:
    """
    Initial condition for the idealized horizontal advection test case.

    """
    if tracer_state_now.qv is None:
        raise ValueError(
            "The initial condition for the linear horizontal advection test case requires the 'qv' to be active."
        )

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    vertex_x = geometry.get(geometry_meta.VERTEX_X).ndarray
    vertex_y = geometry.get(geometry_meta.VERTEX_Y).ndarray
    cell_center_x = geometry.get(geometry_meta.CELL_CENTER_X).ndarray
    cell_center_y = geometry.get(geometry_meta.CELL_CENTER_Y).ndarray

    # density is set to the inverse of the vertical grid spacing, so that the mass flux equals the velocity
    prognostic_state_now.rho.ndarray[:, :] = metrics.get(metrics_meta.INV_DDQZ_Z_FULL).ndarray

    weights, nodes = _prepare_torus_quadratic_quadrature(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        c2v_connectivity=grid.connectivities["C2V"].ndarray,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )

    _fill_prep_adv_from_prescribed_wind_field(
        velocity_field=config.velocity_field,
        prep_adv_state=tracer_prep_adv_state,
        primal_normal_x=geometry.get(geometry_meta.EDGE_NORMAL_U).ndarray,
        primal_normal_y=geometry.get(geometry_meta.EDGE_NORMAL_V).ndarray,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )

    center_x, center_y = _compute_tracer_center(
        initial_center=config.initial_center,
        origin_x=vertex_x.min(),
        origin_y=vertex_y.min(),
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )
    _fill_tracer_from_analytical_profile(
        config=config,
        tracer_buffer=tracer_state_now.qv.ndarray,
        weights=weights,
        nodes=nodes,
        tracer_center_x=center_x,
        tracer_center_y=center_y,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )


def construct_reference_tracer(
    *,
    config: LinearHorizontalAdvectionConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    integration_time: float,
    num_levels: int,
) -> data_alloc.NDArray:
    geometry = static_fields.geometry
    vertex_x = geometry.get(geometry_meta.VERTEX_X).ndarray
    vertex_y = geometry.get(geometry_meta.VERTEX_Y).ndarray
    cell_center_x = geometry.get(geometry_meta.CELL_CENTER_X).ndarray
    cell_center_y = geometry.get(geometry_meta.CELL_CENTER_Y).ndarray

    weights, nodes = _prepare_torus_quadratic_quadrature(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        c2v_connectivity=grid.connectivities["C2V"].ndarray,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )

    array_ns = data_alloc.array_namespace(cell_center_x)
    reference_tracer = array_ns.tile(array_ns.zeros_like(cell_center_x)[:, None], (1, num_levels))
    u, v = _compute_idealized_horizontal_velocity_field(
        velocity_field=config.velocity_field,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )
    end_center_x, end_center_y = _compute_tracer_center(
        initial_center=config.initial_center,
        origin_x=vertex_x.min(),
        origin_y=vertex_y.min(),
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
        displacement_x=u * integration_time,
        displacement_y=v * integration_time,
    )
    _fill_tracer_from_analytical_profile(
        config=config,
        tracer_buffer=reference_tracer,
        weights=weights,
        nodes=nodes,
        tracer_center_x=end_center_x,
        tracer_center_y=end_center_y,
        domain_length=grid.grid_params.domain_length,
        domain_height=grid.grid_params.domain_height,
    )
    return reference_tracer
