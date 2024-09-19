# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import pathlib

import gt4py.next as gtx

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


# TODO (Chia Rui): Convert all numpy computations to cupy
def model_initialization_jabw(
    grid: icon_grid.IconGrid,
    cell_param: geometry.CellParams,
    edge_param: geometry.EdgeParams,
    path: pathlib.Path,
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
    Initial condition of Jablonowski-Williamson test. Set jw_up to values larger than 0.01 if
    you want to run baroclinic case.

    Args:
        grid: IconGrid
        cell_param: cell properties
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

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().ndarray
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().ndarray
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().ndarray
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().ndarray
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().ndarray
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().ndarray
    geopot = data_provider.from_metrics_savepoint().geopot().ndarray

    cell_lat = cell_param.cell_center_lat.ndarray
    edge_lat = edge_param.edge_center[0].ndarray
    edge_lon = edge_param.edge_center[1].ndarray
    primal_normal_x = edge_param.primal_normal[0].ndarray

    cell_2_edge_coeff = data_provider.from_interpolation_savepoint().c_lin_e()
    rbf_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbf_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    num_cells = grid.num_cells
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

    p_sfc = 100000.0
    jw_up = 0.0  # if doing baroclinic wave test, please set it to a nonzero value
    jw_u0 = 35.0
    jw_temp0 = 288.0
    # DEFINED PARAMETERS for jablonowski williamson:
    eta_0 = 0.252
    eta_t = 0.2  # tropopause
    gamma = 0.005  # temperature elapse rate (K/m)
    dtemp = 4.8e5  # empirical temperature difference (K)
    # for baroclinic wave test
    lon_perturbation_center = math.pi / 9.0  # longitude of the perturb centre
    lat_perturbation_center = 2.0 * lon_perturbation_center  # latitude of the perturb centre
    ps_o_p0ref = p_sfc / phy_const.P0REF

    w_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=float)
    exner_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    rho_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    temperature_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    pressure_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    theta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=float)
    eta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=float)

    sin_lat = xp.sin(cell_lat)
    cos_lat = xp.cos(cell_lat)
    fac1 = 1.0 / 6.3 - 2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0)
    fac2 = (
        (8.0 / 5.0 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.25 * math.pi)
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    lapse_rate = phy_const.RD * gamma / phy_const.GRAV
    for k_index in range(num_levels - 1, -1, -1):
        eta_old = xp.full(num_cells, fill_value=1.0e-7, dtype=float)
        log.info(f"In Newton iteration, k = {k_index}")
        # Newton iteration to determine zeta
        for _ in range(100):
            eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = xp.cos(eta_v_ndarray[:, k_index])
            sin_etav = xp.sin(eta_v_ndarray[:, k_index])

            temperature_avg = jw_temp0 * (eta_old**lapse_rate)
            geopot_avg = jw_temp0 * phy_const.GRAV / gamma * (1.0 - eta_old**lapse_rate)
            temperature_avg = xp.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = xp.where(
                eta_old < eta_t,
                geopot_avg
                - phy_const.RD
                * dtemp
                * (
                    (xp.log(eta_old / eta_t) + 137.0 / 60.0) * (eta_t**5)
                    - 5.0 * (eta_t**4) * eta_old
                    + 5.0 * (eta_t**3) * (eta_old**2)
                    - 10.0 / 3.0 * (eta_t**2) * (eta_old**3)
                    + 1.25 * eta_t * (eta_old**4)
                    - 0.2 * (eta_old**5)
                ),
                geopot_avg,
            )

            geopot_jw = geopot_avg + jw_u0 * (cos_etav**1.5) * (
                fac1 * jw_u0 * (cos_etav**1.5) + fac2
            )
            temperature_jw = (
                temperature_avg
                + 0.75
                * eta_old
                * math.pi
                * jw_u0
                / phy_const.RD
                * sin_etav
                * xp.sqrt(cos_etav)
                * (2.0 * jw_u0 * fac1 * (cos_etav**1.5) + fac2)
            )
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -phy_const.RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        # Use analytic expressions at all model level
        exner_ndarray[:, k_index] = (eta_old * ps_o_p0ref) ** phy_const.RD_O_CPD
        theta_v_ndarray[:, k_index] = temperature_jw / exner_ndarray[:, k_index]
        rho_ndarray[:, k_index] = (
            exner_ndarray[:, k_index] ** phy_const.CVD_O_RD
            * phy_const.P0REF
            / phy_const.RD
            / theta_v_ndarray[:, k_index]
        )
        # initialize diagnose pressure and temperature variables
        pressure_ndarray[:, k_index] = (
            phy_const.P0REF * exner_ndarray[:, k_index] ** phy_const.CPD_O_RD
        )
        temperature_ndarray[:, k_index] = temperature_jw
    log.info("Newton iteration completed!")

    eta_v = gtx.as_field((dims.CellDim, dims.KDim), eta_v_ndarray)
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

    print("debug: ", eta_v_e.ndarray.device)
    vn_ndarray = testcases_utils.zonalwind_2_normalwind_ndarray(
        grid,
        jw_u0,
        jw_up,
        lat_perturbation_center,
        lon_perturbation_center,
        edge_lat,
        edge_lon,
        primal_normal_x,
        eta_v_e.ndarray,
    )
    log.info("U2vn computation completed.")

    rho_ndarray, exner_ndarray, theta_v_ndarray = testcases_utils.hydrostatic_adjustment_ndarray(
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

    vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray)
    w = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray)
    exner = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray)
    rho = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray)
    temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_ndarray)
    virutal_temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_ndarray)
    pressure = gtx.as_field((dims.CellDim, dims.KDim), pressure_ndarray)
    theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray)
    pressure_ifc_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=float)
    pressure_ifc_ndarray[:, -1] = p_sfc
    pressure_ifc = gtx.as_field((dims.CellDim, dims.KDim), pressure_ifc_ndarray)

    vn_next = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray)
    w_next = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray)
    exner_next = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray)
    rho_next = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray)
    theta_v_next = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray)

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
        virtual_temperature=virutal_temperature,
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

    diffusion_diagnostic_state = diffus_states.DiffusionDiagnosticState(
        hdef_ic=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid, is_halfdim=True
        ),
        div_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
        dwdx=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
        dwdy=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=grid, is_halfdim=True),
    )
    solve_nonhydro_diagnostic_state = solve_nh_states.DiagnosticStateNonHydro(
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
        exner_dyn_incr_lastsubstep=field_alloc.allocate_zero_field(
            dims.CellDim, dims.KDim, grid=grid
        ),
    )

    prep_adv = solve_nh_states.PrepAdvection(
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
        0.0,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )
