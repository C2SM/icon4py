# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Moving-cylinder tracer advection experiment (Jocksch, FFSL-WENO on a planar torus).

A passive tracer that is 1 inside a cylinder and 0 outside is carried once around a
periodic torus by a uniform wind, with unit air mass and no vertical motion. After one
period the exact solution is the initial field again, so the error is simply the
difference between the final and the initial tracer.

The setup follows the live block in Jocksch's ``mo_nh_stepping.f90``: the tracer is
sampled at the cell centres (1 where the centre distance is strictly less than the
radius), ``mass_flx_me = vn_traj = v . n_e`` with the unit primal normal ``n_e``,
``mass_flx_ic = 0``, ``airmass = 1`` and ``ddqz_z_full = 1``. Here the unit air mass is
obtained through ``rho = 1 / ddqz_z_full`` instead, which is what the driver's airmass
update turns into ``airmass = 1``.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid
from icon4py.model.common.initial_condition.analytical import (
    linear_horizontal_advection as lin_hor_adv_ic,
)
from icon4py.model.common.math import distance_array_ns
from icon4py.model.common.metrics import metrics_attributes as metrics_meta
from icon4py.model.common.states import adv_states, prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.states import static_fields


@dataclasses.dataclass
class MovingCylinderConfig:
    center_x: typing.Annotated[
        float | None,
        common_conf_opt.ConfigOption(
            description="Cylinder centre x-coordinate [m] in the grid file's coordinates; None puts it at the domain centre.",
            icon_equivalent=None,
        ),
    ] = None
    center_y: typing.Annotated[
        float | None,
        common_conf_opt.ConfigOption(
            description="Cylinder centre y-coordinate [m] in the grid file's coordinates; None puts it at the domain centre.",
            icon_equivalent=None,
        ),
    ] = None
    radius: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Cylinder radius [m].",
            icon_equivalent=None,
        ),
    ] = 25000.0
    wind_speed: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Magnitude of the uniform wind [m/s].",
            icon_equivalent=None,
        ),
    ] = 1.0
    wind_angle: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Direction of the uniform wind, counter-clockwise from the x axis [degrees].",
            icon_equivalent=None,
        ),
    ] = 0.0

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_wind_components(config: MovingCylinderConfig) -> tuple[float, float]:
    """The (u, v) components of the uniform wind."""
    angle = math.radians(config.wind_angle)
    return config.wind_speed * math.cos(angle), config.wind_speed * math.sin(angle)


def compute_cylinder_center(
    config: MovingCylinderConfig, domain_length: float, domain_height: float
) -> tuple[float, float]:
    """The cylinder centre, defaulting to the domain centre in the grid file's coordinates."""
    center_x = config.center_x if config.center_x is not None else 0.5 * domain_length
    center_y = config.center_y if config.center_y is not None else 0.5 * domain_height
    return center_x, center_y


def sample_cylinder(
    *,
    config: MovingCylinderConfig,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """The cylinder sampled at the cell centres: 1 strictly inside the radius, 0 elsewhere.

    The distance to the centre is the minimum-image (periodic) one, so the cylinder may
    straddle the domain boundary.
    """
    array_ns = data_alloc.array_namespace(cell_center_x)
    center_x, center_y = compute_cylinder_center(config, domain_length, domain_height)
    dx, dy = distance_array_ns.minimum_image_separation(
        x=cell_center_x,
        y=cell_center_y,
        reference_x=center_x,
        reference_y=center_y,
        domain_extent_x=domain_length,
        domain_extent_y=domain_height,
    )
    return array_ns.where(dx**2 + dy**2 < config.radius**2, 1.0, 0.0)


def moving_cylinder(
    *,
    config: MovingCylinderConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    adv_prep_adv_state: adv_states.AdvectionPrepAdvState,
) -> None:
    """Initial condition for the moving-cylinder tracer advection experiment."""
    if grid.grid_params.geometry_type != icon_grid.GeometryType.TORUS:
        raise NotImplementedError(
            "The 'moving_cylinder' initial condition is only implemented on a torus grid."
        )
    if tracer_state_now.qv is None:
        raise ValueError(
            "The 'moving_cylinder' initial condition requires an active qv tracer (ntracer >= 1)."
        )
    domain_length = grid.grid_params.domain_length
    domain_height = grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None

    geometry = static_fields.geometry
    metrics = static_fields.metrics

    # unit air mass: the driver computes airmass = rho * ddqz_z_full
    prognostic_state_now.rho.ndarray[:, :] = metrics.get(metrics_meta.INV_DDQZ_Z_FULL).ndarray

    u, v = compute_wind_components(config)
    lin_hor_adv_ic.prescribe_uniform_wind(
        prep_adv_state=adv_prep_adv_state,
        primal_normal_x=geometry.get(geometry_meta.EDGE_NORMAL_U).ndarray,
        primal_normal_y=geometry.get(geometry_meta.EDGE_NORMAL_V).ndarray,
        u=u,
        v=v,
    )

    cylinder = sample_cylinder(
        config=config,
        cell_center_x=geometry.get(geometry_meta.CELL_CENTER_X).ndarray,
        cell_center_y=geometry.get(geometry_meta.CELL_CENTER_Y).ndarray,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    tracer_state_now.qv.ndarray[:, :] = cylinder[:, None]
