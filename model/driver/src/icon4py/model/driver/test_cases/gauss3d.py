# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import pathlib

import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion import diffusion_states as diff_states
from icon4py.model.atmosphere.dycore import init_exner_pr
from icon4py.model.atmosphere.dycore.state_utils import (
    states as states_utils,
    utils as dycore_utils,
)
from icon4py.model.common import constants as const
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.grid.horizontal import EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation as c2e_interp,
    edge_2_cell_vector_rbf_interpolation as e2c_interp,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import (
    diagnostic_state as diag_states,
    prognostic_state as prog_states,
)
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.driver.test_cases import utils as testcases_utils


log = logging.getLogger(__name__)


def model_initialization_gauss3d(
    icon_grid: icon_grid.IconGrid,
    edge_param: EdgeParams,
    path: pathlib.Path,
    rank=0,
) -> tuple[
    diff_states.DiffusionDiagnosticState,
    states_utils.DiagnosticStateNonHydro,
    states_utils.PrepAdvection,
    float,
    diag_states.DiagnosticState,
    prog_states.PrognosticState,
    prog_states.PrognosticState,
]:
    """
    Initial condition for the Gauss 3D test.

    Args:
        icon_grid: IconGrid
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

    num_cells = icon_grid.num_cells
    num_edges = icon_grid.num_edges
    num_levels = icon_grid.num_levels

    grid_idx_edge_start_plus1 = icon_grid.get_end_index(
        EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1
    )
    grid_idx_edge_end = icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim))
    grid_idx_cell_interior_start = icon_grid.get_start_index(
        CellDim, HorizontalMarkerIndex.interior(CellDim)
    )
    grid_idx_cell_start_plus1 = icon_grid.get_end_index(
        CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    )
    grid_idx_cell_end = icon_grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))

    w_numpy = xp.zeros((num_cells, num_levels + 1), dtype=float)
    exner_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    rho_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    temperature_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    pressure_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    theta_v_numpy = xp.zeros((num_cells, num_levels), dtype=float)
    eta_v_numpy = xp.zeros((num_cells, num_levels), dtype=float)

    mask_array_edge_start_plus1_to_edge_end = xp.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:grid_idx_edge_start_plus1] = False
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
    vn_numpy = u * primal_normal_x
    log.info("Wind profile assigned.")

    # Vertical temperature profile
    for k_index in range(num_levels - 1, -1, -1):
        z_help = (nh_brunt_vais / const.GRAV) ** 2 * geopot[:, k_index]
        # profile of theta is explicitly given
        theta_v_numpy[:, k_index] = nh_t0 * xp.exp(z_help)

    # Lower boundary condition for exner pressure
    if nh_brunt_vais != 0.0:
        z_help = (nh_brunt_vais / const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_numpy[:, num_levels - 1] = (const.GRAV / nh_brunt_vais) ** 2 / nh_t0 / const.CPD * (
            xp.exp(-z_help) - 1.0
        ) + 1.0
    else:
        exner_numpy[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / const.CPD / nh_t0
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

    eta_v = gtx.as_field((CellDim, KDim), eta_v_numpy)
    eta_v_e = dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid)
    c2e_interp.cell_2_edge_interpolation(
        eta_v,
        cell_2_edge_coeff,
        eta_v_e,
        grid_idx_edge_start_plus1,
        grid_idx_edge_end,
        0,
        num_levels,
        offset_provider=icon_grid.offset_providers,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    vn = gtx.as_field((EdgeDim, KDim), vn_numpy)
    w = gtx.as_field((CellDim, KDim), w_numpy)
    exner = gtx.as_field((CellDim, KDim), exner_numpy)
    rho = gtx.as_field((CellDim, KDim), rho_numpy)
    temperature = gtx.as_field((CellDim, KDim), temperature_numpy)
    pressure = gtx.as_field((CellDim, KDim), pressure_numpy)
    theta_v = gtx.as_field((CellDim, KDim), theta_v_numpy)
    pressure_ifc_numpy = xp.zeros((num_cells, num_levels + 1), dtype=float)
    pressure_ifc_numpy[
        :, -1
    ] = const.P0REF  # set surface pressure to the prescribed value (only used for IC in JABW test case, then actually computed in the dycore)
    pressure_ifc = gtx.as_field((CellDim, KDim), pressure_ifc_numpy)

    vn_next = gtx.as_field((EdgeDim, KDim), vn_numpy)
    w_next = gtx.as_field((CellDim, KDim), w_numpy)
    exner_next = gtx.as_field((CellDim, KDim), exner_numpy)
    rho_next = gtx.as_field((CellDim, KDim), rho_numpy)
    theta_v_next = gtx.as_field((CellDim, KDim), theta_v_numpy)

    u = dycore_utils._allocate(CellDim, KDim, grid=icon_grid)
    v = dycore_utils._allocate(CellDim, KDim, grid=icon_grid)
    e2c_interp.edge_2_cell_vector_rbf_interpolation(
        vn,
        rbf_vec_coeff_c1,
        rbf_vec_coeff_c2,
        u,
        v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
        0,
        num_levels,
        offset_provider=icon_grid.offset_providers,
    )
    log.info("U, V computation completed.")

    exner_pr = dycore_utils._allocate(CellDim, KDim, grid=icon_grid)
    init_exner_pr.init_exner_pr(
        exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        exner_pr,
        grid_idx_cell_interior_start,
        grid_idx_cell_end,
        0,
        num_levels,
        offset_provider={},
    )
    log.info("exner_pr initialization completed.")

    diagnostic_state = diag_states.DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        u=u,
        v=v,
    )

    prognostic_state_now = prog_states.PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = prog_states.PrognosticState(
        w=w_next,
        vn=vn_next,
        theta_v=theta_v_next,
        rho=rho_next,
        exner=exner_next,
    )

    diffusion_diagnostic_state = diff_states.DiffusionDiagnosticState(
        hdef_ic=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        div_ic=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdx=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdy=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
    )
    solve_nonhydro_diagnostic_state = states_utils.DiagnosticStateNonHydro(
        theta_v_ic=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        exner_pr=exner_pr,
        rho_ic=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_exner_phy=dycore_utils._allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_rho=dycore_utils._allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_thv=dycore_utils._allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_w=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        mass_fl_e=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_phy=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        grf_tend_vn=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl1=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl2=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_w_adv_ntl1=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_w_adv_ntl2=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        vt=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        vn_ie=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid, is_halfdim=True),
        w_concorr_c=dycore_utils._allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=dycore_utils._allocate(CellDim, KDim, grid=icon_grid),
    )

    prep_adv = states_utils.PrepAdvection(
        vn_traj=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_me=dycore_utils._allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_ic=dycore_utils._allocate(CellDim, KDim, grid=icon_grid),
        vol_flx_ic=dycore_utils.zero_field(icon_grid, CellDim, KDim, dtype=float),
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
