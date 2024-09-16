# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging


from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.test_utils import helpers, serialbox_utils as sb
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# flake8: noqa
log = logging.getLogger(__name__)


def construct_config():
    return advection.AdvectionConfig()


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
    icon_grid, deepatmo: bool = False
) -> advection_states.AdvectionMetricState:
    if deepatmo:
        raise NotImplementedError("Data for deep atmosphere runs has not been serialized yet.")
    return advection_states.AdvectionMetricState(
        deepatmo_divh=helpers.constant_field(icon_grid, 1.0, dims.KDim),
        deepatmo_divzl=constant_field(icon_grid, 1.0, dims.KDim),
        deepatmo_divzu=constant_field(icon_grid, 1.0, dims.KDim),
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
    return advection_states.AdvectionDiagnosticState(
        airmass_now=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=icon_grid
        ),  # init field
        airmass_new=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=icon_grid
        ),  # init field
        grf_tend_tracer=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=icon_grid
        ),  # init field
        hfl_tracer=savepoint.hfl_tracer(ntracer),
        vfl_tracer=savepoint.vfl_tracer(ntracer),
    )


def construct_prep_adv(
    icon_grid, savepoint: sb.AdvectionInitSavepoint
) -> solve_nh_states.PrepAdvection:
    return solve_nh_states.PrepAdvection(
        vn_traj=savepoint.vn_traj(),
        mass_flx_me=savepoint.mass_flx_me(),
        mass_flx_ic=savepoint.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=icon_grid
        ),  # unused in advection
    )


def log_dbg(field, name=""):
    log.debug(f"{name}: min={field.min()}, max={field.max()}, mean={field.mean()}")


def log_serialized(
    diagnostic_state: advection_states.AdvectionDiagnosticState,
    prep_adv: solve_nh_states.PrepAdvection,
    p_tracer_now: fa.CellKField[wpfloat],
    dtime: wpfloat,
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
    p_tracer_new: fa.CellKField[wpfloat],
    p_tracer_new_ref: fa.CellKField[wpfloat],
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
