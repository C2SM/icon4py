# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum, auto
import dataclasses
import logging

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.test_utils import helpers, serialbox_utils as sb
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# flake8: noqa
log = logging.getLogger(__name__)


class InitialConditions(Enum):
    """
    Initial conditions for idealized advection test cases.
    """

    #: one-dimensional smooth Gaussian function
    GAUSSIAN_1D = auto()

    #: one-dimensional discontinuous function
    BOX_1D = auto()


class VelocityField(Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: constant velocity field
    CONSTANT_POSITIVE = auto()
    CONSTANT_NEGATIVE = auto()

    #: time-dependent velocity field
    TEMPORAL_COS = auto()

    #: space-dependent velocity field
    SPATIAL_PARABOLA = auto()
    SPATIAL_SIN = auto()


@dataclasses.dataclass(frozen=True)
class TestConfig:
    initial_conditions: InitialConditions
    velocity_field: VelocityField

    """
    Contains necessary parameters to configure an advection test case.
    """

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        if not hasattr(InitialConditions, self.initial_conditions.name):
            raise NotImplementedError(
                f"Initial conditions type {self.initial_conditions} not implemented."
            )
        if not hasattr(VelocityField, self.velocity_field.name):
            raise NotImplementedError(f"Velocity Field type {self.velocity_field} not implemented.")


def construct_test_config(
    initial_conditions: InitialConditions,
    velocity_field: VelocityField = VelocityField.CONSTANT_POSITIVE,
) -> TestConfig:
    return TestConfig(initial_conditions=initial_conditions, velocity_field=velocity_field)


def construct_config(
    horizontal_advection_type: advection.HorizontalAdvectionType,
    horizontal_advection_limiter: advection.HorizontalAdvectionLimiter,
    vertical_advection_type: advection.VerticalAdvectionType,
    vertical_advection_limiter: advection.VerticalAdvectionLimiter,
) -> advection.AdvectionConfig:
    return advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )


def construct_interpolation_state(
    savepoint: sb.InterpolationSavepoint,
) -> advection_states.AdvectionInterpolationState:
    return advection_states.AdvectionInterpolationState(
        geofac_div=helpers.as_1D_sparse_field(savepoint.geofac_div(), dims.CEDim),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        pos_on_tplane_e_1=savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=savepoint.pos_on_tplane_e_y(),
    )


def construct_least_squares_state(
    savepoint: sb.LeastSquaresSavepoint,
) -> advection_states.AdvectionLeastSquaresState:
    return advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=savepoint.lsq_pseudoinv_1(),
        lsq_pseudoinv_2=savepoint.lsq_pseudoinv_2(),
    )


def construct_metric_state(
    icon_grid, savepoint: sb.MetricSavepoint
) -> advection_states.AdvectionMetricState:
    constant_f = helpers.constant_field(icon_grid, 1.0, dims.KDim)
    ddqz_z_full_np = np.reciprocal(savepoint.inv_ddqz_z_full().asnumpy())
    return advection_states.AdvectionMetricState(
        deepatmo_divh=constant_f,
        deepatmo_divzl=constant_f,
        deepatmo_divzu=constant_f,
        ddqz_z_full=gtx.as_field((dims.CellDim, dims.KDim), ddqz_z_full_np),
    )


def construct_diagnostic_init_state(
    icon_grid, savepoint: sb.AdvectionInitSavepoint, ntracer: int
) -> advection_states.AdvectionDiagnosticState:
    return advection_states.AdvectionDiagnosticState(
        airmass_now=savepoint.airmass_now(),
        airmass_new=savepoint.airmass_new(),
        grf_tend_tracer=savepoint.grf_tend_tracer(ntracer),
        hfl_tracer=field_alloc.allocate_zero_field(
            dims.EdgeDim, dims.KDim, grid=icon_grid
        ),  # exit field
        vfl_tracer=field_alloc.allocate_zero_field(  # TODO (dastrm): should be KHalfDim
            dims.CellDim, dims.KDim, is_halfdim=True, grid=icon_grid
        ),  # exit field
    )


def construct_diagnostic_exit_state(
    icon_grid, savepoint: sb.AdvectionInitSavepoint, ntracer: int
) -> advection_states.AdvectionDiagnosticState:
    zero_f = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid)
    return advection_states.AdvectionDiagnosticState(
        airmass_now=zero_f,  # init field
        airmass_new=zero_f,  # init field
        grf_tend_tracer=zero_f,  # init field
        hfl_tracer=savepoint.hfl_tracer(ntracer),
        vfl_tracer=savepoint.vfl_tracer(ntracer),
    )


def construct_prep_adv(
    icon_grid, savepoint: sb.AdvectionInitSavepoint
) -> advection_states.AdvectionPrepAdvState:
    return advection_states.AdvectionPrepAdvState(
        vn_traj=savepoint.vn_traj(),
        mass_flx_me=savepoint.mass_flx_me(),
        mass_flx_ic=savepoint.mass_flx_ic(),
    )


def construct_idealized_diagnostic_state(
    icon_grid, ddqz_z_full
) -> advection_states.AdvectionDiagnosticState:
    return advection_states.AdvectionDiagnosticState(
        airmass_now=ddqz_z_full,
        airmass_new=ddqz_z_full,
        grf_tend_tracer=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
        hfl_tracer=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=icon_grid),
        vfl_tracer=field_alloc.allocate_zero_field(  # TODO (dastrm): should be KHalfDim
            dims.CellDim, dims.KDim, is_halfdim=True, grid=icon_grid
        ),
    )


def construct_idealized_prep_adv(
    test_config,
    icon_grid,
    z_ifc,
    z_range,
    time,
    dtime,
    time_end,
) -> advection_states.AdvectionPrepAdvState:
    # impose 1D velocity field at time n+1/2 as required by the numerical scheme
    w = get_idealized_velocity_field(test_config, z_range, z_ifc, time + dtime / 2.0, time_end)
    log_dbg(w, "w")

    w = np.repeat(np.expand_dims(w, axis=-1), icon_grid.num_cells, axis=1).T

    zero_f = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid)
    return advection_states.AdvectionPrepAdvState(
        vn_traj=zero_f,
        mass_flx_me=zero_f,  # v_n*rho*dz
        mass_flx_ic=gtx.as_field((dims.CellDim, dims.KDim), w),  # w*rho
    )


def construct_idealized_metric_state(
    icon_grid, ddqz_z_full
) -> advection_states.AdvectionMetricState:
    constant_f = helpers.constant_field(icon_grid, 1.0, dims.KDim)
    return advection_states.AdvectionMetricState(
        deepatmo_divh=constant_f,
        deepatmo_divzl=constant_f,
        deepatmo_divzu=constant_f,
        ddqz_z_full=ddqz_z_full,
    )


def construct_idealized_tracer(
    test_config,
    icon_grid,
    z_mc,
    z_ifc,
    z_center,
    z_range,
    use_high_order_quadrature,
) -> fa.CellKField[ta.wpfloat]:
    # impose tracer ICs at the horizontal grid center
    z = z_mc - z_center
    tracer = get_idealized_ICs(test_config, z, z_range)
    if use_high_order_quadrature:
        # Simpson's 1/3 rule
        tracer_ifc = get_idealized_ICs(test_config, z_ifc - z_center, z_range)
        tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
    log_dbg(tracer, "tracer")

    tracer = np.repeat(np.expand_dims(tracer, axis=-1), icon_grid.num_cells, axis=1).T
    return gtx.as_field((dims.CellDim, dims.KDim), tracer)


def construct_idealized_tracer_reference(
    test_config,
    icon_grid,
    z_mc,
    z_center,
    z_range,
    z_ifc,
    time,
    time_end,
    use_high_order_quadrature,
    reference_solution=None,
    z_mc_high=None,
) -> fa.CellKField[ta.wpfloat]:
    if reference_solution is None:
        # use exact solution
        match test_config.velocity_field:
            case VelocityField.CONSTANT_POSITIVE | VelocityField.CONSTANT_NEGATIVE:
                # linearly shifted ICs
                w_mc = get_idealized_velocity_field(test_config, z_range, z_mc, time, time_end)
                tracer = get_idealized_ICs(test_config, z_mc - (z_center + w_mc * time), z_range)
                if use_high_order_quadrature:
                    # Simpson's 1/3 rule
                    w_ifc = get_idealized_velocity_field(
                        test_config, z_range, z_ifc, time, time_end
                    )
                    tracer_ifc = get_idealized_ICs(
                        test_config, z_ifc - (z_center + w_ifc * time), z_range
                    )
                    tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
            case VelocityField.TEMPORAL_COS:
                # shifted ICs
                w_mc = get_idealized_velocity_field(test_config, z_range, z_mc, time, time_end)
                shift = z_center + (2 * z_center) * np.sin(np.pi * time) / np.pi
                tracer = get_idealized_ICs(test_config, z_mc - shift, z_range)
                if use_high_order_quadrature:
                    # Simpson's 1/3 rule
                    tracer_ifc = get_idealized_ICs(test_config, z_ifc - shift, z_range)
                    tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
            case VelocityField.SPATIAL_PARABOLA:
                # shifted and deformed ICs
                w_mc = get_idealized_velocity_field(test_config, z_range, z_mc, time, time_end)
                z = z_range * z_mc / (z_range + time * z_mc)
                fac = (z_range / np.abs(z_range + time * z_mc)) ** 2
                tracer = fac * get_idealized_ICs(test_config, z - z_center, z_range)
                if use_high_order_quadrature:
                    # Simpson's 1/3 rule
                    z = z_range * z_ifc / (z_range + time * z_ifc)
                    fac = (z_range / np.abs(z_range + time * z_ifc)) ** 2
                    tracer_ifc = fac * get_idealized_ICs(test_config, z - z_center, z_range)
                    tracer = (tracer_ifc[:-1] + 4.0 * tracer + tracer_ifc[1:]) / 6.0
            case _:
                raise NotImplementedError(
                    f"Exact solution with velocity field {test_config.velocity_field} not implemented."
                )
    else:
        # use high-resolution numerical solution
        jc = 0
        reference_solution_jc = reference_solution.ndarray[jc, :]
        num_levels = np.size(z_mc)
        num_levels_ref = np.size(z_mc_high)
        assert num_levels_ref >= num_levels
        assert num_levels_ref % num_levels == 0
        n = num_levels_ref // num_levels
        log.debug(f"reference resolution factor n={n}")

        # interpolate
        tracer = reference_solution_jc.reshape(num_levels, -1).mean(axis=1)

    log_dbg(tracer, "tracer")

    tracer = np.repeat(np.expand_dims(tracer, axis=-1), icon_grid.num_cells, axis=1).T
    return gtx.as_field((dims.CellDim, dims.KDim), tracer)


def get_idealized_ICs(test_config, z, z_range):
    match test_config.initial_conditions:
        case InitialConditions.GAUSSIAN_1D:
            s = z_range ** (-1.5)
            return np.exp(-s * (z**2))
        case InitialConditions.BOX_1D:
            r = z_range / 8.0
            return np.where(z**2 <= r**2, 1.0, 0.0)
        case _:
            raise NotImplementedError(
                f"Initial conditions {test_config.initial_conditions} not implemented."
            )


def get_idealized_velocity_field(test_config, z_range, z_arr, time, time_end):
    # note: assumes that time is at n+1/2
    match test_config.velocity_field:
        case VelocityField.CONSTANT_POSITIVE:
            w = z_range * np.ones_like(z_arr)
        case VelocityField.CONSTANT_NEGATIVE:
            w = -z_range * np.ones_like(z_arr)
        case VelocityField.TEMPORAL_COS:
            w = z_range * np.ones_like(z_arr) * np.cos(np.pi * time)
        case VelocityField.SPATIAL_PARABOLA:
            w = z_arr * z_arr / z_range
        case VelocityField.SPATIAL_SIN:
            w = z_range * np.sin(np.pi * z_arr / z_range)
        case _:
            raise NotImplementedError(
                f"Velocity field {test_config.velocity_field} not implemented."
            )
    return w


def get_idealized_velocity_max(test_config, z_range, time_end):
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max

    match test_config.velocity_field:
        case (
            VelocityField.CONSTANT_POSITIVE
            | VelocityField.CONSTANT_NEGATIVE
            | VelocityField.TEMPORAL_COS
            | VelocityField.SPATIAL_PARABOLA
            | VelocityField.SPATIAL_SIN
        ):
            vel_max = z_range
        case _:
            raise NotImplementedError(
                f"Velocity field {test_config.velocity_field} not implemented."
            )
    return vel_max


def compute_relative_errors(values, reference):
    # compute the errors relative to the reference

    # note: the following lines take the errors of all the cells, which is fine
    if isinstance(values, gtx.common.Field):
        values = values.ndarray
    if isinstance(reference, gtx.common.Field):
        reference = reference.ndarray

    error_l1 = np.sum(np.abs(values - reference)) / np.sum(np.abs(reference))
    error_linf = np.max(np.abs(values - reference)) / np.max(np.abs(reference))

    return error_l1, error_linf


def log_dbg(field, name=""):
    log.debug(f"{name}: min={field.min()}, max={field.max()}, mean={field.mean()}")


def log_serialized(
    diagnostic_state: advection_states.AdvectionDiagnosticState,
    prep_adv: advection_states.AdvectionPrepAdvState,
    p_tracer_now: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
):
    log_dbg(diagnostic_state.airmass_now.asnumpy(), "airmass_now")
    log_dbg(diagnostic_state.airmass_new.asnumpy(), "airmass_new")
    log_dbg(diagnostic_state.grf_tend_tracer.asnumpy(), "grf_tend_tracer")
    log_dbg(prep_adv.vn_traj.asnumpy(), "vn_traj")
    log_dbg(prep_adv.mass_flx_me.asnumpy(), "mass_flx_me")
    log_dbg(prep_adv.mass_flx_ic.asnumpy(), "mass_flx_ic")
    log_dbg(p_tracer_now.asnumpy(), "p_tracer_now")
    log.debug(f"dtime: {dtime}")


def verify_advection_fields(
    grid: icon_grid.IconGrid,
    diagnostic_state: advection_states.AdvectionDiagnosticState,
    diagnostic_state_ref: advection_states.AdvectionDiagnosticState,
    p_tracer_new: fa.CellKField[ta.wpfloat],
    p_tracer_new_ref: fa.CellKField[ta.wpfloat],
    even_timestep: bool,
):
    # cell indices
    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
    start_cell_lateral_boundary_level_2 = grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    # edge indices
    edge_domain = h_grid.domain(dims.EdgeDim)
    start_edge_lateral_boundary_level_5 = grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )
    end_edge_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))

    hfl_tracer_range = np.arange(start_edge_lateral_boundary_level_5, end_edge_halo)
    vfl_tracer_range = (
        np.arange(start_cell_lateral_boundary_level_2, end_cell_end)
        if even_timestep
        else np.arange(start_cell_nudging, end_cell_local)
    )
    p_tracer_new_range = np.arange(start_cell_lateral_boundary, end_cell_local)

    # log advection output fields
    log_dbg(diagnostic_state.hfl_tracer.asnumpy()[hfl_tracer_range, :], "hfl_tracer")
    log_dbg(diagnostic_state_ref.hfl_tracer.asnumpy()[hfl_tracer_range, :], "hfl_tracer_ref")
    log_dbg(diagnostic_state.vfl_tracer.asnumpy()[vfl_tracer_range, :], "vfl_tracer")
    log_dbg(diagnostic_state_ref.vfl_tracer.asnumpy()[vfl_tracer_range, :], "vfl_tracer_ref")
    log_dbg(p_tracer_new.asnumpy()[p_tracer_new_range, :], "p_tracer_new")
    log_dbg(p_tracer_new_ref.asnumpy()[p_tracer_new_range, :], "p_tracer_new_ref")

    # verify advection output fields
    assert helpers.dallclose(
        diagnostic_state.hfl_tracer.asnumpy()[hfl_tracer_range, :],
        diagnostic_state_ref.hfl_tracer.asnumpy()[hfl_tracer_range, :],
        rtol=1e-10,
    )
    assert helpers.dallclose(
        diagnostic_state.vfl_tracer.asnumpy()[vfl_tracer_range, :],
        diagnostic_state_ref.vfl_tracer.asnumpy()[vfl_tracer_range, :],
        rtol=1e-10,
    )
    assert helpers.dallclose(
        p_tracer_new.asnumpy()[p_tracer_new_range, :],
        p_tracer_new_ref.asnumpy()[p_tracer_new_range, :],
        atol=1e-16,
    )
