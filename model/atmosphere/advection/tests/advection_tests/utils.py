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

from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    torus_helpers,
    helpers,
    serialbox_utils as sb,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# flake8: noqa
log = logging.getLogger(__name__)


class InitialConditions(Enum):
    """
    Initial conditions for idealized advection test cases.
    """

    #: two-dimensional smooth Gaussian curve
    GAUSSIAN_2D = auto()
    #: two-dimensional smooth off-centered Gaussian curve
    GAUSSIAN_2D_OFFCENTER = auto()
    #: two-dimensional discontinuous circle
    CIRCLE_2D = auto()


class VelocityField(Enum):
    """
    Velocity field for idealized advection test cases.
    """

    #: constant velocity field
    CONSTANT = auto()
    #: two-dimensional divergence-free swirling velocity field
    VORTEX_2D = auto()
    #: two-dimensional increasingly deformational field
    INCREASING_2D = auto()


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
    initial_conditions: InitialConditions, velocity_field: VelocityField = VelocityField.CONSTANT
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


def construct_metric_state(icon_grid) -> advection_states.AdvectionMetricState:
    constant_f = helpers.constant_field(icon_grid, 1.0, dims.KDim)
    return advection_states.AdvectionMetricState(
        deepatmo_divh=constant_f,
        deepatmo_divzl=constant_f,
        deepatmo_divzu=constant_f,
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


def construct_idealized_diagnostic_state(icon_grid) -> advection_states.AdvectionDiagnosticState:
    return advection_states.AdvectionDiagnosticState(
        airmass_now=helpers.constant_field(icon_grid, 1.0, dims.CellDim, dims.KDim),
        airmass_new=helpers.constant_field(icon_grid, 1.0, dims.CellDim, dims.KDim),
        grf_tend_tracer=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
        hfl_tracer=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=icon_grid),
        vfl_tracer=field_alloc.allocate_zero_field(  # TODO (dastrm): should be KHalfDim
            dims.CellDim, dims.KDim, is_halfdim=True, grid=icon_grid
        ),
    )


def construct_idealized_prep_adv(
    test_config,
    icon_grid,
    edge_geometry,
    edges_center_x,
    edges_center_y,
    x_range,
    y_range,
    time,
    dtime,
    time_end,
) -> advection_states.AdvectionPrepAdvState:
    # note: since we assume that the airmass is constant 1.0, the mass flux equals the velocity

    primal_normal_x = edge_geometry.primal_normal[0].ndarray
    primal_normal_y = edge_geometry.primal_normal[1].ndarray

    # impose 2D velocity field at time n+1/2 as required by the numerical scheme
    u, v = get_idealized_velocity_field(
        test_config, x_range, y_range, edges_center_x, edges_center_y, time + dtime / 2.0, time_end
    )
    vn = u * primal_normal_x + v * primal_normal_y
    log_dbg(vn, "vn")

    vn = xp.repeat(xp.expand_dims(vn, axis=-1), icon_grid.num_levels, axis=1)

    return advection_states.AdvectionPrepAdvState(
        vn_traj=gtx.as_field((dims.EdgeDim, dims.KDim), vn),
        mass_flx_me=gtx.as_field((dims.EdgeDim, dims.KDim), vn),
        mass_flx_ic=field_alloc.allocate_zero_field(  # TODO (dastrm): should be KHalfDim
            dims.CellDim, dims.KDim, is_halfdim=True, grid=icon_grid
        ),
    )


def construct_idealized_tracer(
    test_config,
    icon_grid,
    x_center,
    y_center,
    x_range,
    y_range,
    weights,
    nodes,
) -> fa.CellKField[ta.wpfloat]:
    # impose tracer ICs at the horizontal grid center
    x = nodes[0, :, :] - x_center
    y = nodes[1, :, :] - y_center
    tracer = xp.sum(weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0)
    log_dbg(tracer, "tracer")

    tracer = xp.repeat(xp.expand_dims(tracer, axis=-1), icon_grid.num_levels, axis=1)
    return gtx.as_field((dims.CellDim, dims.KDim), tracer)


def construct_idealized_tracer_reference(
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
    time,
    time_end,
    weights,
    nodes,
    reference_solution=None,
    cell_center_x_high=None,
    cell_center_y_high=None,
) -> fa.CellKField[ta.wpfloat]:
    if reference_solution is None:
        # use exact solution
        match test_config.velocity_field:
            case VelocityField.CONSTANT:
                # linearly shifted ICs
                u, v = get_idealized_velocity_field(
                    test_config, x_range, y_range, edges_center_x, edges_center_y, time, time_end
                )
                x = nodes[0, :, :] - (x_center + u * time)
                y = nodes[1, :, :] - (y_center + v * time)
                tracer = xp.sum(
                    weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
                )
            case VelocityField.VORTEX_2D:
                # ICs
                x = nodes[0, :, :] - x_center
                y = nodes[1, :, :] - y_center
                tracer = xp.sum(
                    weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
                )
            case VelocityField.INCREASING_2D:
                # shifted and deformed ICs
                et = xp.exp(time)
                emt = xp.exp(-time)
                x = -emt * (-x_range + x_range * et - x_range * time - nodes[0, :, :]) - x_center
                y = -y_range + y_range * et - y_range * et * time + nodes[1, :, :] * et - y_center
                tracer = xp.sum(
                    weights * get_idealized_ICs(test_config, x, y, x_range, y_range), axis=0
                )
            case _:
                raise NotImplementedError(
                    f"Exact solution with velocity field {test_config.velocity_field} not implemented."
                )
    else:
        # use high-resolution numerical solution
        k = 0
        reference_solution_k = reference_solution.ndarray[:, k]
        tracer = torus_helpers.interpolate_torus_plane(
            cell_center_x_high,
            cell_center_y_high,
            reference_solution_k,
            node_x,
            node_y,
            weights,
            nodes,
        )

    log_dbg(tracer, "tracer")

    tracer = xp.repeat(xp.expand_dims(tracer, axis=-1), icon_grid.num_levels, axis=1)
    return gtx.as_field((dims.CellDim, dims.KDim), tracer)


def get_idealized_ICs(test_config, x, y, x_range, y_range):
    match test_config.initial_conditions:
        case InitialConditions.GAUSSIAN_2D:
            s = ((x_range + y_range) / 2.0) ** (-1.65)
            return xp.exp(-s * (x**2 + y**2))
        case InitialConditions.GAUSSIAN_2D_OFFCENTER:
            y -= y_range / 4
            s = ((x_range + y_range) / 2.0) ** (-1.5)
            return xp.exp(-s * (x**2 + y**2))
        case InitialConditions.CIRCLE_2D:
            r = (x_range + y_range) / 8.0
            return xp.where(x**2 + y**2 <= r**2, 1.0, 0.0)
        case _:
            raise NotImplementedError(
                f"Initial conditions {test_config.initial_conditions} not implemented."
            )


def get_idealized_velocity_field(
    test_config, x_range, y_range, edges_center_x, edges_center_y, time, time_end
):
    # note: assumes that time is at n+1/2
    match test_config.velocity_field:
        case VelocityField.CONSTANT:
            u, v = x_range, y_range
        case VelocityField.VORTEX_2D:
            v_scal = 1e0  # measure for deformation, originally 1e0
            u = (
                -v_scal
                * x_range
                * (xp.sin(xp.pi * edges_center_x / x_range) ** 2)
                * xp.cos(xp.pi * edges_center_y / y_range)
                * xp.sin(xp.pi * edges_center_y / y_range)
                * xp.cos(xp.pi * time / time_end)
            )
            v = (
                v_scal
                * y_range
                * xp.cos(xp.pi * edges_center_x / x_range)
                * xp.sin(xp.pi * edges_center_x / x_range)
                * (xp.sin(xp.pi * edges_center_y / y_range) ** 2)
                * xp.cos(xp.pi * time / time_end)
            )
        case VelocityField.INCREASING_2D:
            u = edges_center_x + x_range * time
            v = -edges_center_y + y_range * time
        case _:
            raise NotImplementedError(
                f"Velocity field {test_config.velocity_field} not implemented."
            )
    return u, v


def get_idealized_velocity_max(test_config, x_range, y_range, time_end):
    # note: as we need vel_max at time n+1/2 and vel_max is needed for the time step, we have a chicken-and-egg problem
    # instead of doing a fixed-point iteration, we simply estimate an upper bound for vel_max

    match test_config.velocity_field:
        case VelocityField.CONSTANT | VelocityField.VORTEX_2D:
            vel_max = (x_range**2 + y_range**2) ** 0.5
        case VelocityField.INCREASING_2D:
            vel_max = (
                (x_range + x_range * time_end) ** 2 + (y_range + y_range * time_end) ** 2
            ) ** 0.5
        case _:
            raise NotImplementedError(
                f"Velocity field {test_config.velocity_field} not implemented."
            )
    return vel_max


def compute_relative_errors(values, reference):
    # compute the errors relative to the reference

    # note: the following lines take the errors of all the levels, which is fine
    if isinstance(values, gtx.common.Field):
        values = values.ndarray
    if isinstance(reference, gtx.common.Field):
        reference = reference.ndarray

    error_l1 = xp.sum(xp.abs(values - reference)) / xp.sum(xp.abs(reference))
    error_linf = xp.max(xp.abs(values - reference)) / xp.max(xp.abs(reference))

    return error_l1, error_linf


def prepare_torus_quadrature(
    icon_grid,
    node_x,
    node_y,
    cell_center_x,
    cell_center_y,
    length_min,
    use_high_order_quadrature=True,
):
    if use_high_order_quadrature:
        weights, nodes = torus_helpers.prepare_torus_quadratic_quadrature(
            icon_grid, node_x, node_y, cell_center_x, cell_center_y, length_min
        )
    else:
        # use cell centers for one-point quadrature rule
        weights = xp.ones_like(cell_center_x).reshape((1, -1))
        nodes = xp.stack((cell_center_x, cell_center_y)).reshape((2, 1, -1))

    return weights, nodes


def get_torus_dimensions(experiment):
    match experiment:
        case (
            dt_utils.TORUS_CONVERGENCE_EXPERIMENT_1
            | dt_utils.TORUS_CONVERGENCE_EXPERIMENT_2
            | dt_utils.TORUS_CONVERGENCE_EXPERIMENT_3
            | dt_utils.TORUS_CONVERGENCE_EXPERIMENT_4
            | dt_utils.TORUS_CONVERGENCE_EXPERIMENT_5
        ):
            # grid dimensions need to be hardcoded such that each grid solves the same problem
            # note: they cannot be inferred from the grid file, only from what is used in the grid generator
            x_center, y_center = 5e4, 5e4
            x_range, y_range = 1e5, 1e5
        case _:
            raise NotImplementedError(f"Unknown grid dimensions for experiment {experiment}.")
    return x_center, y_center, x_range, y_range


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
):
    # cell indices
    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
    end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    # edge indices
    edge_domain = h_grid.domain(dims.EdgeDim)
    start_edge_lateral_boundary_level_5 = grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )
    end_edge_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))

    # log advection output fields
    log_dbg(
        diagnostic_state.hfl_tracer.asnumpy()[start_edge_lateral_boundary_level_5:end_edge_halo, :],
        "hfl_tracer",
    )
    log_dbg(
        diagnostic_state_ref.hfl_tracer.asnumpy()[
            start_edge_lateral_boundary_level_5:end_edge_halo, :
        ],
        "hfl_tracer_ref",
    )
    log_dbg(
        diagnostic_state.vfl_tracer.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        "vfl_tracer",
    )
    log_dbg(
        diagnostic_state_ref.vfl_tracer.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        "vfl_tracer_ref",
    )
    log_dbg(p_tracer_new.asnumpy()[start_cell_lateral_boundary:end_cell_local, :], "p_tracer_new")
    log_dbg(
        p_tracer_new_ref.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        "p_tracer_new_ref",
    )

    # verify advection output fields
    assert helpers.dallclose(
        diagnostic_state.hfl_tracer.asnumpy()[start_edge_lateral_boundary_level_5:end_edge_halo, :],
        diagnostic_state_ref.hfl_tracer.asnumpy()[
            start_edge_lateral_boundary_level_5:end_edge_halo, :
        ],
        rtol=1e-10,
    )
    assert helpers.dallclose(  # TODO (dastrm): adjust indices once there is vertical transport
        diagnostic_state.vfl_tracer.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        diagnostic_state_ref.vfl_tracer.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        rtol=1e-10,
    )
    assert helpers.dallclose(
        p_tracer_new.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        p_tracer_new_ref.asnumpy()[start_cell_lateral_boundary:end_cell_local, :],
        atol=1e-16,
    )
