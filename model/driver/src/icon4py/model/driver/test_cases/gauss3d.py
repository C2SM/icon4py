# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import pathlib

import gt4py.next as gtx
from gt4py.next import backend as gt4py_backend

from icon4py.model.atmosphere.diffusion import diffusion_states as diffus_states
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import geometry, horizontal as h_grid, icon as icon_grid
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation,
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.driver.test_cases import utils as testcases_utils


log = logging.getLogger(__name__)


def model_initialization_gauss3d(
    grid: icon_grid.IconGrid,
    edge_param: geometry.EdgeParams,
    path: pathlib.Path,
    backend: gt4py_backend.Backend,
    rank=0,
) -> tuple[
    diffus_states.DiffusionDiagnosticState,
    solve_nh_states.DiagnosticStateNonHydro,
    solve_nh_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Initial condition for the Gauss 3D test.

    Args:
        grid: IconGrid
        edge_param: edge properties
        path: path where to find the input data
        backend: GT4Py backend
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().ndarray
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().ndarray
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().ndarray
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().ndarray
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().ndarray
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().ndarray
    geopot = data_provider.from_metrics_savepoint().geopot().ndarray

    primal_normal_x = edge_param.primal_normal[0].ndarray

    cell_2_edge_coeff = data_provider.from_interpolation_savepoint().c_lin_e()
    rbf_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbf_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    num_cells = grid.num_cells
    num_edges = grid.num_edges
    num_levels = grid.num_levels

    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    end_edge_lateral_boundary_level_2 = grid.end_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_edge_end = grid.end_index(edge_domain(h_grid.Zone.END))
    end_cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    w_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=float)
    exner_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    rho_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    temperature_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    pressure_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    theta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    eta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=float)

    mask_array_edge_start_plus1_to_edge_end = xp.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:end_edge_lateral_boundary_level_2] = False
    mask = xp.repeat(
        xp.expand_dims(mask_array_edge_start_plus1_to_edge_end, axis=-1), num_levels, axis=1
    )
    primal_normal_x = xp.repeat(xp.expand_dims(primal_normal_x, axis=-1), num_levels, axis=1)

    # Define test case parameters
    # The topography can only be read from serialized data for now, then these
    # variables should be defined here and used to compute the idealized
    # topography:
    # - mount_lon
    # - mount_lat
    # - mount_height
    # - mount_width
    nh_t0 = 300.0
    nh_u0 = 0.0
    nh_brunt_vais = 0.01
    log.info("Topography can only be read from serialized data for now.")

    # Horizontal wind field
    u = xp.where(mask, nh_u0, 0.0)
    vn_ndarray = u * primal_normal_x
    log.info("Wind profile assigned.")

    # Vertical temperature profile
    for k_index in range(num_levels - 1, -1, -1):
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        # profile of theta is explicitly given
        theta_v_ndarray[:, k_index] = nh_t0 * xp.exp(z_help)

    # Lower boundary condition for exner pressure
    if nh_brunt_vais != 0.0:
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / nh_brunt_vais
        ) ** 2 / nh_t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / nh_t0
    log.info("Vertical computations completed.")

    # Compute hydrostatically balanced exner, by integrating the (discretized!)
    # 3rd equation of motion under the assumption thetav=const.
    rho_ndarray, exner_ndarray = testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho_ndarray,
        exner_ndarray,
        theta_v_ndarray,
        num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    eta_v = gtx.as_field((dims.CellDim, dims.KDim), eta_v_ndarray)
    eta_v_e = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid)
    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        eta_v,
        cell_2_edge_coeff,
        eta_v_e,
        end_edge_lateral_boundary_level_2,
        end_edge_end,
        0,
        num_levels,
        offset_provider=grid.offset_providers,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    pressure_ifc_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=float)
    pressure_ifc_ndarray[
        :, -1
    ] = phy_const.P0REF  # set surface pressure to the prescribed value (only used for IC in JABW test case, then actually computed in the dycore)
    (
        vn,
        w,
        exner,
        rho,
        theta_v,
        vn_next,
        w_next,
        exner_next,
        rho_next,
        theta_v_next,
        temperature,
        virtual_temperature,
        pressure,
        pressure_ifc,
        u,
        v,
    ) = testcases_utils.create_gt4py_field_for_prognostic_and_diagnostic_variables(
        vn_ndarray,
        w_ndarray,
        exner_ndarray,
        rho_ndarray,
        theta_v_ndarray,
        temperature_ndarray,
        pressure_ndarray,
        pressure_ifc_ndarray,
        grid=grid,
        backend=backend,
    )

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation(
        vn,
        rbf_vec_coeff_c1,
        rbf_vec_coeff_c2,
        u,
        v,
        end_cell_lateral_boundary_level_2,
        end_cell_end,
        0,
        num_levels,
        offset_provider=grid.offset_providers,
    )
    log.info("U, V computation completed.")

    exner_pr = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid)
    testcases_utils.compute_perturbed_exner.with_backend(backend)(
        exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        exner_pr,
        0,
        num_cells,
        0,
        num_levels,
        offset_provider={},
    )
    log.info("exner_pr initialization completed.")

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        u=u,
        v=v,
    )

    prognostic_state_now = prognostics.PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = prognostics.PrognosticState(
        w=w_next,
        vn=vn_next,
        theta_v=theta_v_next,
        rho=rho_next,
        exner=exner_next,
    )

    diffusion_diagnostic_state = testcases_utils.initialize_diffusion_diagnostic_state(
        grid=grid, backend=backend
    )
    solve_nonhydro_diagnostic_state = testcases_utils.initialize_solve_nonhydro_diagnostic_state(
        exner_pr=exner_pr, grid=grid, backend=backend
    )

    prep_adv = testcases_utils.initialize_prep_advection(grid=grid, backend=backend)
    log.info("Initialization completed.")

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        0.0,  # divdamp_fac_o2 only != 0 for data assimilation
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )
