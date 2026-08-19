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
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.metrics import metrics_attributes as metrics_meta
from icon4py.model.common.states import (
    prep_adv_states,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.metrics import metrics_factory


@config_io.register_enum
class VerticalTracerProfile(int, enum.Enum):
    """
    Initial conditions for idealized advection test cases.
    """

    #: one-dimensional smooth Gaussian function
    GAUSSIAN = 1

    #: one-dimensional discontinuous function
    BOX = 2


@config_io.register_enum
class VerticalVelocityField(int, enum.Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: uniform velocity field
    UNIFORM_POSITIVE = 1
    UNIFORM_NEGATIVE = 2


@dataclasses.dataclass
class LinearVerticalAdvectionConfig:
    tracer_profile: typing.Annotated[
        VerticalTracerProfile,
        common_conf_opt.ConfigOption(
            description="Initial tracer profile.",
            icon_equivalent=None,
        ),
    ] = VerticalTracerProfile.GAUSSIAN
    velocity_field: typing.Annotated[
        VerticalVelocityField,
        common_conf_opt.ConfigOption(
            description="Velocity field for transporting the tracer.",
            icon_equivalent=None,
        ),
    ] = VerticalVelocityField.UNIFORM_POSITIVE
    cfl_number: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum CFL number for determination of the time step.",
            icon_equivalent=None,
        ),
    ] = 0.8
    initial_center: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Initial height of the tracer profile center relative to the model top height.",
            icon_equivalent=None,
        ),
    ] = 0.5
    decay_radius: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Decay radius for the Gaussian tracer profile (0.001 fraction) relative to the model top height.",
            icon_equivalent=None,
        ),
    ] = 0.25

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_max_velocity(
    *,
    velocity_field: VerticalVelocityField,
    model_top_height: float,
) -> float:
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max

    w = _compute_idealized_vertical_velocity_field(
        velocity_field=velocity_field,
        model_top_height=model_top_height,
    )
    return data_alloc.array_namespace(w).max(data_alloc.array_namespace(w).abs(w))


def _compute_idealized_vertical_velocity_field(
    *,
    velocity_field: VerticalVelocityField,
    model_top_height: float,
) -> float:
    # note: assumes that time is at n+1/2
    match velocity_field:
        case VerticalVelocityField.UNIFORM_POSITIVE:
            w = model_top_height
        case VerticalVelocityField.UNIFORM_NEGATIVE:
            w = -model_top_height
        case _:
            raise NotImplementedError(f"Velocity field {velocity_field} not implemented.")
    return w


def _fill_prep_adv_from_prescribed_wind_field(
    *,
    velocity_field: VerticalVelocityField,
    prep_adv_state: prep_adv_states.TracerPrepAdvState,
    model_top_height: float,
) -> None:
    # impose 1D velocity field at time n+1/2 as required by the numerical scheme
    w = _compute_idealized_vertical_velocity_field(
        velocity_field=velocity_field,
        model_top_height=model_top_height,
    )

    vn_traj = prep_adv_state.vn_traj.ndarray
    mass_flx_me = prep_adv_state.mass_flx_me.ndarray
    mass_flx_ic = prep_adv_state.mass_flx_ic.ndarray

    vn_traj[:, :] = 0.0
    mass_flx_me[:, :] = 0.0
    mass_flx_ic[:, :] = w


def _fill_tracer_from_analytical_profile(
    *,
    config: LinearVerticalAdvectionConfig,
    tracer_buffer: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    center_z: float,
    model_top_height: float,
) -> None:
    """
    Create an idealized tracer vertical profile. The profile is constructed using Simpson's
    1/3 rule to integrate the tracer values over neighboring cell interfaces. The accuracy
    of the tracer profile is third order.

    Args:
        config: configuration for the linear vertical advection test case
        tracer_buffer: buffer array to store the tracer values
        z_mc: cell center z-coordinate
        z_ifc: cell interface z-coordinate
        center_z: domain size in x-direction
        model_top_height: height of the model top
    """
    array_ns = data_alloc.array_namespace(z_mc)

    def _compute_tracer(dz: data_alloc.NDArray) -> data_alloc.NDArray:
        match config.tracer_profile:
            case VerticalTracerProfile.GAUSSIAN:
                decay_factor = (
                    -1.0 / (config.decay_radius * model_top_height) ** (2) * math.log(1e-3)
                )
                return array_ns.exp(-decay_factor * (dz**2))
            case VerticalTracerProfile.BOX:
                r = model_top_height / 8.0
                return array_ns.where(dz**2 <= r**2, 1.0, 0.0)
            case _:
                raise NotImplementedError(
                    f"Initial tracer profile {config.tracer_profile} not implemented."
                )

    # Simpson's 1/3 rule
    tracer_mc = _compute_tracer(z_mc - center_z)
    tracer_ifc = _compute_tracer(z_ifc - center_z)
    tracer_buffer[:, :] = (tracer_ifc[:, :-1] + 4.0 * tracer_mc + tracer_ifc[:, 1:]) / 6.0


def linear_vertical_advection(
    *,
    config: LinearVerticalAdvectionConfig,
    vertical_config: v_grid.VerticalGridConfig,
    metrics: metrics_factory.MetricsFieldsFactory,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    tracer_prep_adv_state: prep_adv_states.TracerPrepAdvState,
) -> None:
    """
    Initial condition for the idealized vertical advection test case.

    """
    if tracer_state_now.qv is None:
        raise ValueError(
            "The initial condition for the linear vertical advection test case requires the 'qv' to be active."
        )

    z_mc = metrics.get(metrics_meta.Z_MC).ndarray
    z_ifc = metrics.get(metrics_meta.CELL_HEIGHT_ON_HALF_LEVEL).ndarray

    prognostic_state_now.rho.ndarray[:, :] = 1.0

    _fill_prep_adv_from_prescribed_wind_field(
        velocity_field=config.velocity_field,
        prep_adv_state=tracer_prep_adv_state,
        model_top_height=vertical_config.model_top_height,
    )

    _fill_tracer_from_analytical_profile(
        config=config,
        tracer_buffer=tracer_state_now.qv.ndarray,
        z_mc=z_mc,
        z_ifc=z_ifc,
        center_z=config.initial_center * vertical_config.model_top_height,
        model_top_height=vertical_config.model_top_height,
    )


def construct_reference_tracer(
    *,
    config: LinearVerticalAdvectionConfig,
    metrics: metrics_factory.MetricsFieldsFactory,
    vertical_config: v_grid.VerticalGridConfig,
    integration_time: float,
) -> data_alloc.NDArray:
    z_mc = metrics.get(metrics_meta.Z_MC).ndarray
    z_ifc = metrics.get(metrics_meta.CELL_HEIGHT_ON_HALF_LEVEL).ndarray
    array_ns = data_alloc.array_namespace(z_mc)
    reference_tracer = array_ns.zeros_like(z_mc)
    w = _compute_idealized_vertical_velocity_field(
        velocity_field=config.velocity_field,
        model_top_height=vertical_config.model_top_height,
    )
    end_center_z = config.initial_center * vertical_config.model_top_height + integration_time * w
    _fill_tracer_from_analytical_profile(
        config=config,
        tracer_buffer=reference_tracer,
        z_mc=z_mc,
        z_ifc=z_ifc,
        center_z=end_center_z,
        model_top_height=vertical_config.model_top_height,
    )
    return reference_tracer
