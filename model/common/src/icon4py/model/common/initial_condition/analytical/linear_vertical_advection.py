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
import typing
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common.config import config_io, options as common_conf_opt
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid, vertical as v_grid
from icon4py.model.common.metrics import metrics_attributes as metrics_meta
from icon4py.model.common.states import adv_states, prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.metrics import metrics_factory


@config_io.register_enum
class TracerProfile(int, enum.Enum):
    """
    Initial conditions for idealized advection test cases.
    """

    #: one-dimensional smooth Gaussian function
    GAUSSIAN_1D = 1

    #: one-dimensional discontinuous function
    BOX_1D = 2


@config_io.register_enum
class VelocityField(int, enum.Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: constant velocity field
    CONSTANT = 1

    #: space-dependent velocity field
    SPATIAL_PARABOLA = 2
    SPATIAL_SIN = 3

    #: time-dependent velocity field
    TEMPORAL_COS = 4


@dataclasses.dataclass
class LinearVerticalAdvectionConfig:
    tracer_profile: typing.Annotated[
        TracerProfile,
        common_conf_opt.ConfigOption(
            description="Initial tracer profile.",
            icon_equivalent=None,
        ),
    ] = TracerProfile.GAUSSIAN_1D
    velocity_field: typing.Annotated[
        VelocityField,
        common_conf_opt.ConfigOption(
            description="Velocity field for transporting the tracer.",
            icon_equivalent=None,
        ),
    ] = VelocityField.CONSTANT
    cfl_number: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum CFL number for determination of the time step.",
            icon_equivalent=None,
        ),
    ] = 0.8

    fortran_name_map: ClassVar[dict[str, str]] = {}


def compute_max_velocity(
    *,
    velocity_field: VelocityField,
    z_ifc: data_alloc.NDArray,
    model_top_height: float,
) -> float:
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max

    w = _compute_idealized_velocity_field(
        velocity_field=velocity_field,
        z_ifc=z_ifc,
        model_top_height=model_top_height,
    )
    return data_alloc.array_namespace(w).max(data_alloc.array_namespace(w).abs(w))


def _compute_idealized_velocity_field(
    *,
    velocity_field: VelocityField,
    z_ifc: data_alloc.NDArray,
    model_top_height: float,
) -> data_alloc.NDArray:
    # note: assumes that time is at n+1/2
    array_ns = data_alloc.array_namespace(z_ifc)
    match velocity_field:
        case VelocityField.CONSTANT:
            w = model_top_height * array_ns.ones_like(z_ifc)
        case VelocityField.CONSTANT_NEGATIVE:
            w = -model_top_height * array_ns.ones_like(z_ifc)
        case VelocityField.SPATIAL_PARABOLA:
            w = z_ifc * z_ifc / model_top_height * array_ns.ones_like(z_ifc)
        case VelocityField.SPATIAL_SIN:
            w = model_top_height * array_ns.sin(array_ns.pi * z_ifc / model_top_height)
        case _:
            raise NotImplementedError(
                f"Velocity field {velocity_field} not implemented."
            )
    return w


def _construct_idealized_prep_adv(
    *,
    velocity_field: VelocityField,
    prep_adv_state: adv_states.AdvectionPrepAdvState,
    z_ifc: data_alloc.NDArray,
    model_top_height: float,
) -> None:
    # impose 1D velocity field at time n+1/2 as required by the numerical scheme
    w = _compute_idealized_velocity_field(
        velocity_field=velocity_field,
        model_top_height=model_top_height,
        z_ifc=z_ifc,
    )
    
    vn_traj = prep_adv_state.vn_traj.ndarray
    mass_flx_me = prep_adv_state.mass_flx_me.ndarray
    mass_flx_ic = prep_adv_state.mass_flx_ic.ndarray
    
    vn_traj[:, :] = 0.0
    mass_flx_me[:, :] = 0.0
    mass_flx_ic[:, :] = w[None, :]


def _construct_idealized_tracer(
    tracer_profile: TracerProfile,
    tracer: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    center_z: float,
    model_top_height: float,
) -> None:
    # impose tracer ICs at the horizontal grid center
    array_ns = data_alloc.array_namespace(z_mc)
    def _compute_tracer(dz: data_alloc.NDArray) -> data_alloc.NDArray:
        match tracer_profile:
            case TracerProfile.GAUSSIAN_1D:
                s = model_top_height ** (-1.5)
                return array_ns.exp(-s * (dz**2))
            case TracerProfile.BOX_1D:
                r = model_top_height / 8.0
                return array_ns.where(dz**2 <= r**2, 1.0, 0.0)
            case _:
                raise NotImplementedError(
                    f"Initial tracer profile {tracer_profile} not implemented."
                )
    # Simpson's 1/3 rule
    tracer_mc = _compute_tracer(z_mc - center_z)
    tracer_ifc = _compute_tracer(z_ifc - center_z)
    tracer[:,:] = (tracer_ifc[:,:-1] + 4.0 * tracer_mc + tracer_ifc[:,1:]) / 6.0


def linear_vertical_advection(
    *,
    config: LinearVerticalAdvectionConfig,
    vertical_config: v_grid.VerticalGridConfig,
    metrics: metrics_factory.MetricsFieldsFactory,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    adv_prep_adv_state: adv_states.AdvectionPrepAdvState,
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

    prognostic_state_now.rho.ndarray[:, :] = metrics.get(metrics_meta.INV_DDQZ_Z_FULL).ndarray

    _construct_idealized_prep_adv(
        velocity_field=config.velocity_field,
        prep_adv_state=adv_prep_adv_state,
        model_top_height=vertical_config.model_top_height,
    )

    _construct_idealized_tracer(
        tracer_profile=config.tracer_profile,
        tracer=tracer_state_now.qv.ndarray,
        z_mc=z_mc,
        z_ifc=z_ifc,
        center_z=vertical_config.model_top_height / 2.0,
        model_top_height=vertical_config.model_top_height,
    )


def construct_reference_tracer(
    *,
    velocity_field: VelocityField,
    tracer_profile: TracerProfile,
    metrics: metrics_factory.MetricsFieldsFactory,
    center_z: float,
    model_top_height: float,
    integration_time: float,
    num_levels: int,
) -> data_alloc.NDArray:
    z_mc = metrics.get(metrics_meta.Z_MC).ndarray
    z_ifc = metrics.get(metrics_meta.CELL_HEIGHT_ON_HALF_LEVEL).ndarray
    array_ns = data_alloc.array_namespace(z_mc)
    reference_tracer = array_ns.tile(array_ns.zeros_like(z_mc)[:, None], (1, num_levels))
    # match velocity_field:
    #     case VelocityField.CONSTANT:
    #         w_mc = _compute_idealized_velocity_field(
    #             velocity_field=velocity_field,
    #             model_top_height=model_top_height,
    #             z_ifc=z_ifc,
    #         )
    #         tracer = _construct_idealized_tracer(
    #             tracer_profile=config.tracer_profile,
    #             tracer=tracer_state_now.qv.ndarray,
    #             z_mc=z_mc,
    #             z_ifc=z_ifc,
    #             center_z=vertical_config.model_top_height / 2.0,
    #             model_top_height=vertical_config.model_top_height,
    #         )
    #             # test_config, z_mc - (z_center + w_mc * time), z_range)
    #         # Simpson's 1/3 rule
    #         w_ifc = _compute_idealized_velocity_field(
    #             test_config, z_range, z_ifc, time, time_end
    #         )
    #         tracer_ifc = get_idealized_ICs(
    #             test_config, z_ifc - (z_center + w_ifc * time), z_range
    #         )
    #         tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
    #     case VelocityField.SPATIAL_PARABOLA:
    #         # shifted and deformed ICs
    #         w_mc = _compute_idealized_velocity_field(test_config, z_range, z_mc, time, time_end)
    #         z = z_range * z_mc / (z_range + time * z_mc)
    #         fac = (z_range / np.abs(z_range + time * z_mc)) ** 2
    #         tracer = fac * get_idealized_ICs(test_config, z - z_center, z_range)
    #         # Simpson's 1/3 rule
    #         z = z_range * z_ifc / (z_range + time * z_ifc)
    #         fac = (z_range / np.abs(z_range + time * z_ifc)) ** 2
    #         tracer_ifc = fac * get_idealized_ICs(test_config, z - z_center, z_range)
    #         tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
    #     case _:
    #         raise NotImplementedError(
    #             f"Exact solution with velocity field {velocity_field} not implemented."
    #         )

    return reference_tracer