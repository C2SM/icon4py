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

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, states as grid_states
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
    edge_param: grid_states.EdgeParams,
    path: pathlib.Path,
    rank=0,
) -> tuple[
    diffusion_states.DiffusionDiagnosticState,
    dycore_states.DiagnosticStateNonHydro,
    dycore_states.PrepAdvection,
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
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().asnumpy()
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().asnumpy()
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().asnumpy()
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().asnumpy()
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().asnumpy()
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().asnumpy()
    geopot = data_provider.from_metrics_savepoint().geopot().asnumpy()

    primal_normal_x = edge_param.primal_normal[0].asnumpy()

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

    w_numpy = xp.zeros((num_cells, num_levels + 1), dtype=float)
    exner_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    rho_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    temperature_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    pressure_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    theta_v_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    eta_v_numpy = xp.zeros((num_cells, num_levels), dtype=float)

    mask_array_edge_start_plus1_to_edge_end = xp.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:end_edge_lateral_boundary_level_2] = False
    mask = xp.repeat(
        xp.expand_dims(xp.asarray(mask_array_edge_start_plus1_to_edge_end), axis=-1),
        num_levels,
        axis=1,
    )
    primal_normal_x = xp.repeat(
        xp.expand_dims(xp.asarray(primal_normal_x), axis=-1), num_levels, axis=1
    )

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
    vn_numpy = u * primal_normal_x
    log.info("Wind profile assigned.")

    # Vertical temperature profile
    for k_index in range(num_levels - 1, -1, -1):
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        # profile of theta is explicitly given
        theta_v_numpy[:, k_index] = nh_t0 * xp.exp(z_help)

    # Lower boundary condition for exner pressure
    if nh_brunt_vais != 0.0:
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_numpy[:, num_levels - 1] = (
            phy_const.GRAV / nh_brunt_vais
        ) ** 2 / nh_t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
    else:
        exner_numpy[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / nh_t0
    log.info("Vertical computations completed.")

    # Compute hydrostatically balanced exner, by integrating the (discretized!)
    # 3rd equation of motion under the assumption thetav=const.
    rho_numpy, exner_numpy = testcases_utils.hydrostatic_adjustment_constant_thetav_numpy(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho_numpy,
        exner_numpy,
        theta_v_numpy,
        num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    eta_v = gtx.as_field((dims.CellDim, dims.KDim), eta_v_numpy)
    eta_v_e = field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid)
    cell_2_edge_interpolation.cell_2_edge_interpolation(
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

    vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_numpy)
    w = gtx.as_field((dims.CellDim, dims.KDim), w_numpy)
    exner = gtx.as_field((dims.CellDim, dims.KDim), exner_numpy)
    rho = gtx.as_field((dims.CellDim, dims.KDim), rho_numpy)
    temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_numpy)
    virtual_temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_numpy)
    pressure = gtx.as_field((dims.CellDim, dims.KDim), pressure_numpy)
    theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_numpy)
    pressure_ifc_numpy = xp.zeros((num_cells, num_levels + 1), dtype=float)
    pressure_ifc_numpy[
        :, -1
    ] = phy_const.P0REF  # set surface pressure to the prescribed value (only used for IC in JABW test case, then actually computed in the dycore)
    pressure_ifc = gtx.as_field((dims.CellDim, dims.KDim), pressure_ifc_numpy)

    vn_next = gtx.as_field((dims.EdgeDim, dims.KDim), vn_numpy)
    w_next = gtx.as_field((dims.CellDim, dims.KDim), w_numpy)
    exner_next = gtx.as_field((dims.CellDim, dims.KDim), exner_numpy)
    rho_next = gtx.as_field((dims.CellDim, dims.KDim), rho_numpy)
    theta_v_next = gtx.as_field((dims.CellDim, dims.KDim), theta_v_numpy)

    u = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid)
    v = field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid)
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
    testcases_utils.compute_perturbed_exner(
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

    diffusion_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        div_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
        dwdx=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
        dwdy=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
    )
    solve_nonhydro_diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        theta_v_ic=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        exner_pr=exner_pr,
        rho_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
        ddt_exner_phy=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
        grf_tend_rho=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
        grf_tend_thv=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
        grf_tend_w=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        mass_fl_e=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        ddt_vn_phy=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        grf_tend_vn=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        ddt_vn_apc_ntl1=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        ddt_vn_apc_ntl2=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        ddt_w_adv_ntl1=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        ddt_w_adv_ntl2=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        vt=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        vn_ie=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid, is_halfdim=True),
        w_concorr_c=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
    )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        mass_flx_me=field_alloc.allocate_zero_field(dims.EdgeDim, dims.KDim, grid=grid),
        mass_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
        vol_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid),
    )
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
