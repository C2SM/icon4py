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
import math
from enum import Enum
from pathlib import Path

import numpy as np
from gt4py.next import as_field
from gt4py.next.common import Field

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.dycore.init_exner_pr import init_exner_pr
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    InterpolationState,
    MetricStateNonHydro,
    PrepAdvection,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate, zero_field
from icon4py.model.common.constants import (
    CPD_O_RD,
    CVD_O_RD,
    EARTH_ANGULAR_VELOCITY,
    EARTH_RADIUS,
    GRAV,
    P0REF,
    RD,
    RD_O_CPD,
)
from icon4py.model.common.decomposition.definitions import DecompositionInfo, ProcessProperties
from icon4py.model.common.decomposition.mpi_decomposition import ParallelLogger
from icon4py.model.common.dimension import (
    CEDim,
    CellDim,
    EdgeDim,
    KDim,
)
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.states.diagnostic_state import DiagnosticMetricState, DiagnosticState
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field
from icon4py.model.driver.jablonowski_willamson_testcase import zonalwind_2_normalwind_jabw_numpy
from icon4py.model.driver.serialbox_helpers import (
    construct_diagnostics_for_diffusion,
    construct_interpolation_state_for_diffusion,
    construct_metric_state_for_diffusion,
)
from icon4py.model.driver.testcase_functions import hydrostatic_adjustment_numpy


SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."
INITIALIZATION_ERROR_MSG = (
    "Only ANY (read from serialized data) and JABW are implemented for model initialization."
)

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"


class ExperimentType(str, Enum):
    JABW = "jabw"
    """initial condition of Jablonowski-Williamson test"""
    ANY = "any"
    """any test with initial conditions read from serialized data (remember to set correct SIMULATION_START_DATE)"""


def read_icon_grid(
    path: Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_root=2,
    grid_level=4,
) -> IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        rank: mpi rank of the current compute node
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads
        from ppser serialized test data
        grid_root: global grid root division number
        grid_level: global grid refinement number
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid(grid_root, grid_level)
            .construct_icon_grid(on_gpu=False)
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def model_initialization_jabw(
    icon_grid: IconGrid,
    cell_param: CellParams,
    edge_param: EdgeParams,
    path: Path,
    rank=0,
) -> tuple[
    DiffusionDiagnosticState,
    DiagnosticStateNonHydro,
    PrepAdvection,
    float,
    DiagnosticState,
    PrognosticState,
    PrognosticState,
]:
    """
    Initial condition of Jablonowski-Williamson test. Set jw_up to values larger than 0.01 if
    you want to run baroclinic case.

    Args:
        icon_grid: IconGrid
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

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().asnumpy()
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().asnumpy()
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().asnumpy()
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().asnumpy()
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().asnumpy()
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().asnumpy()
    geopot = data_provider.from_metrics_savepoint().geopot().asnumpy()

    cell_lat = cell_param.cell_center_lat.asnumpy()
    edge_lat = edge_param.edge_center[0].asnumpy()
    edge_lon = edge_param.edge_center[1].asnumpy()
    primal_normal_x = edge_param.primal_normal[0].asnumpy()

    cell_2_edge_coeff = data_provider.from_interpolation_savepoint().c_lin_e()
    rbf_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbf_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    cell_size = cell_lat.size
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
    ps_o_p0ref = p_sfc / P0REF

    w_numpy = np.zeros((cell_size, num_levels + 1), dtype=float)
    exner_numpy = np.zeros((cell_size, num_levels), dtype=float)
    rho_numpy = np.zeros((cell_size, num_levels), dtype=float)
    temperature_numpy = np.zeros((cell_size, num_levels), dtype=float)
    pressure_numpy = np.zeros((cell_size, num_levels), dtype=float)
    theta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)
    eta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)

    sin_lat = np.sin(cell_lat)
    cos_lat = np.cos(cell_lat)
    fac1 = 1.0 / 6.3 - 2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0)
    fac2 = (
        (8.0 / 5.0 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.25 * math.pi)
        * EARTH_RADIUS
        * EARTH_ANGULAR_VELOCITY
    )
    lapse_rate = RD * gamma / GRAV
    for k_index in range(num_levels - 1, -1, -1):
        eta_old = np.full(cell_size, fill_value=1.0e-7, dtype=float)
        log.info(f"In Newton iteration, k = {k_index}")
        # Newton iteration to determine zeta
        for _ in range(100):
            eta_v_numpy[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = np.cos(eta_v_numpy[:, k_index])
            sin_etav = np.sin(eta_v_numpy[:, k_index])

            temperature_avg = jw_temp0 * (eta_old**lapse_rate)
            geopot_avg = jw_temp0 * GRAV / gamma * (1.0 - eta_old**lapse_rate)
            temperature_avg = np.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = np.where(
                eta_old < eta_t,
                geopot_avg
                - RD
                * dtemp
                * (
                    (np.log(eta_old / eta_t) + 137.0 / 60.0) * (eta_t**5)
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
                / RD
                * sin_etav
                * np.sqrt(cos_etav)
                * (2.0 * jw_u0 * fac1 * (cos_etav**1.5) + fac2)
            )
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v_numpy[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        # Use analytic expressions at all model level
        exner_numpy[:, k_index] = (eta_old * ps_o_p0ref) ** RD_O_CPD
        theta_v_numpy[:, k_index] = temperature_jw / exner_numpy[:, k_index]
        rho_numpy[:, k_index] = (
            exner_numpy[:, k_index] ** CVD_O_RD * P0REF / RD / theta_v_numpy[:, k_index]
        )
        # initialize diagnose pressure and temperature variables
        pressure_numpy[:, k_index] = P0REF * exner_numpy[:, k_index] ** CPD_O_RD
        temperature_numpy[:, k_index] = temperature_jw
    log.info("Newton iteration completed!")

    eta_v = as_field((CellDim, KDim), eta_v_numpy)
    eta_v_e = _allocate(EdgeDim, KDim, grid=icon_grid)
    cell_2_edge_interpolation(
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

    vn_numpy = zonalwind_2_normalwind_jabw_numpy(
        icon_grid,
        jw_u0,
        jw_up,
        lat_perturbation_center,
        lon_perturbation_center,
        edge_lat,
        edge_lon,
        primal_normal_x,
        eta_v_e.asnumpy(),
    )
    log.info("U2vn computation completed.")

    rho_numpy, exner_numpy, theta_v_numpy = hydrostatic_adjustment_numpy(
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

    vn = as_field((EdgeDim, KDim), vn_numpy)
    w = as_field((CellDim, KDim), w_numpy)
    exner = as_field((CellDim, KDim), exner_numpy)
    rho = as_field((CellDim, KDim), rho_numpy)
    temperature = as_field((CellDim, KDim), temperature_numpy)
    pressure = as_field((CellDim, KDim), pressure_numpy)
    theta_v = as_field((CellDim, KDim), theta_v_numpy)
    pressure_ifc_numpy = np.zeros((cell_size, num_levels + 1), dtype=float)
    pressure_ifc_numpy[:, -1] = p_sfc
    pressure_ifc = as_field((CellDim, KDim), pressure_ifc_numpy)

    vn_next = as_field((EdgeDim, KDim), vn_numpy)
    w_next = as_field((CellDim, KDim), w_numpy)
    exner_next = as_field((CellDim, KDim), exner_numpy)
    rho_next = as_field((CellDim, KDim), rho_numpy)
    theta_v_next = as_field((CellDim, KDim), theta_v_numpy)

    u = _allocate(CellDim, KDim, grid=icon_grid)
    v = _allocate(CellDim, KDim, grid=icon_grid)
    edge_2_cell_vector_rbf_interpolation(
        vn,
        rbf_vec_coeff_c1,
        rbf_vec_coeff_c2,
        u,
        v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
        0,
        icon_grid.num_levels,
        offset_provider=icon_grid.offset_providers,
    )

    log.info("U, V computation completed.")

    exner_pr = _allocate(CellDim, KDim, grid=icon_grid)
    init_exner_pr(
        exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        exner_pr,
        grid_idx_cell_interior_start,
        grid_idx_cell_end,
        0,
        icon_grid.num_levels,
        offset_provider={},
    )
    log.info("exner_pr initialization completed.")

    diagnostic_state = DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        u=u,
        v=v,
    )

    prognostic_state_now = PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = PrognosticState(
        w=w_next,
        vn=vn_next,
        theta_v=theta_v_next,
        rho=rho_next,
        exner=exner_next,
    )

    diffusion_diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        div_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdx=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        dwdy=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
    )
    solve_nonhydro_diagnostic_state = DiagnosticStateNonHydro(
        theta_v_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        exner_pr=exner_pr,
        rho_ic=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_exner_phy=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_rho=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_thv=_allocate(CellDim, KDim, grid=icon_grid),
        grf_tend_w=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        mass_fl_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_phy=_allocate(EdgeDim, KDim, grid=icon_grid),
        grf_tend_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl1=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_vn_apc_ntl2=_allocate(EdgeDim, KDim, grid=icon_grid),
        ddt_w_adv_ntl1=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        ddt_w_adv_ntl2=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        vt=_allocate(EdgeDim, KDim, grid=icon_grid),
        vn_ie=_allocate(EdgeDim, KDim, grid=icon_grid, is_halfdim=True),
        w_concorr_c=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=_allocate(CellDim, KDim, grid=icon_grid),
    )

    prep_adv = PrepAdvection(
        vn_traj=_allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_me=_allocate(EdgeDim, KDim, grid=icon_grid),
        mass_flx_ic=_allocate(CellDim, KDim, grid=icon_grid),
        vol_flx_ic=zero_field(icon_grid, CellDim, KDim, dtype=float),
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


def model_initialization_serialbox(
    icon_grid: IconGrid, path: Path, rank=0
) -> tuple[
    DiffusionDiagnosticState,
    DiagnosticStateNonHydro,
    PrepAdvection,
    float,
    DiagnosticState,
    PrognosticState,
    PrognosticState,
]:
    """
    Initial condition read from serialized data. Diagnostic variables are allocated as zero
    fields.

    Args:
        icon_grid: IconGrid
        path: path where to find the input data
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )
    diffusion_init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    solve_nonhydro_init_savepoint = data_provider.from_savepoint_nonhydro_init(
        istep=1, date=SIMULATION_START_DATE, jstep=0
    )
    velocity_init_savepoint = data_provider.from_savepoint_velocity_init(
        istep=1, vn_only=False, date=SIMULATION_START_DATE, jstep=0
    )
    prognostic_state_now = diffusion_init_savepoint.construct_prognostics()
    diffusion_diagnostic_state = construct_diagnostics_for_diffusion(
        diffusion_init_savepoint,
    )
    solve_nonhydro_diagnostic_state = DiagnosticStateNonHydro(
        theta_v_ic=solve_nonhydro_init_savepoint.theta_v_ic(),
        exner_pr=solve_nonhydro_init_savepoint.exner_pr(),
        rho_ic=solve_nonhydro_init_savepoint.rho_ic(),
        ddt_exner_phy=solve_nonhydro_init_savepoint.ddt_exner_phy(),
        grf_tend_rho=solve_nonhydro_init_savepoint.grf_tend_rho(),
        grf_tend_thv=solve_nonhydro_init_savepoint.grf_tend_thv(),
        grf_tend_w=solve_nonhydro_init_savepoint.grf_tend_w(),
        mass_fl_e=solve_nonhydro_init_savepoint.mass_fl_e(),
        ddt_vn_phy=solve_nonhydro_init_savepoint.ddt_vn_phy(),
        grf_tend_vn=solve_nonhydro_init_savepoint.grf_tend_vn(),
        ddt_vn_apc_ntl1=velocity_init_savepoint.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=velocity_init_savepoint.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=velocity_init_savepoint.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=velocity_init_savepoint.ddt_w_adv_pc(2),
        vt=velocity_init_savepoint.vt(),
        vn_ie=velocity_init_savepoint.vn_ie(),
        w_concorr_c=velocity_init_savepoint.w_concorr_c(),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=None,
    )

    diagnostic_state = DiagnosticState(
        pressure=_allocate(CellDim, KDim, grid=icon_grid),
        pressure_ifc=_allocate(CellDim, KDim, grid=icon_grid, is_halfdim=True),
        temperature=_allocate(CellDim, KDim, grid=icon_grid),
        u=_allocate(CellDim, KDim, grid=icon_grid),
        v=_allocate(CellDim, KDim, grid=icon_grid),
    )

    prognostic_state_next = PrognosticState(
        w=solve_nonhydro_init_savepoint.w_new(),
        vn=solve_nonhydro_init_savepoint.vn_new(),
        theta_v=solve_nonhydro_init_savepoint.theta_v_new(),
        rho=solve_nonhydro_init_savepoint.rho_new(),
        exner=solve_nonhydro_init_savepoint.exner_new(),
    )

    prep_adv = PrepAdvection(
        vn_traj=solve_nonhydro_init_savepoint.vn_traj(),
        mass_flx_me=solve_nonhydro_init_savepoint.mass_flx_me(),
        mass_flx_ic=solve_nonhydro_init_savepoint.mass_flx_ic(),
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        solve_nonhydro_init_savepoint.divdamp_fac_o2(),
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_initial_state(
    icon_grid: IconGrid,
    cell_param: CellParams,
    edge_param: EdgeParams,
    path: Path,
    rank=0,
    experiment_type: ExperimentType = ExperimentType.ANY,
) -> tuple[
    DiffusionDiagnosticState,
    DiagnosticStateNonHydro,
    PrepAdvection,
    float,
    DiagnosticState,
    PrognosticState,
    PrognosticState,
]:
    """
    Read initial prognostic and diagnostic fields.

    Args:
        icon_grid: IconGrid
        cell_param: cell properties
        edge_param: edge properties
        path: path to the serialized input data
        rank: mpi rank of the current compute node
        experiment_type: (optional) defaults to ANY=any, type of initial condition to be read

    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    if experiment_type == ExperimentType.JABW:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_jabw(icon_grid, cell_param, edge_param, path, rank)
    elif experiment_type == ExperimentType.ANY:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_serialbox(icon_grid, path, rank)
    else:
        raise NotImplementedError(INITIALIZATION_ERROR_MSG)

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_geometry_fields(
    path: Path,
    damping_height,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_root=2,
    grid_level=4,
) -> tuple[EdgeParams, CellParams, VerticalModelParams, Field[[CellDim], bool]]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        damping_height: damping height for Rayleigh and divergence damping
        rank: mpi rank of the current compute node
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read
        grid_root: global grid root division number
        grid_level: global grid refinement number

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        ).from_savepoint_grid(grid_root, grid_level)
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vertical_geometry = VerticalModelParams(
            vct_a=sp.vct_a(),
            rayleigh_damping_height=damping_height,
            nflatlev=sp.nflatlev(),
            nflat_gradp=sp.nflat_gradp(),
        )
        return edge_geometry, cell_geometry, vertical_geometry, sp.c_owner_mask()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_decomp_info(
    path: Path,
    procs_props: ProcessProperties,
    ser_type=SerializationType.SB,
    grid_root=2,
    grid_level=4,
) -> DecompositionInfo:
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), True, procs_props.rank
        )
        return sp.from_savepoint_grid(grid_root, grid_level).construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    path: Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_root=2,
    grid_level=4,
) -> tuple[
    DiffusionMetricState,
    DiffusionInterpolationState,
    MetricStateNonHydro,
    InterpolationState,
    DiagnosticMetricState,
]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read
        grid_root: global grid root division number
        grid_level: global grid refinement number

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        data_provider = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        )
        icon_grid = (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid(grid_root, grid_level)
            .construct_icon_grid(on_gpu=False)
        )
        diffusion_interpolation_state = construct_interpolation_state_for_diffusion(
            data_provider.from_interpolation_savepoint()
        )
        diffusion_metric_state = construct_metric_state_for_diffusion(
            data_provider.from_metrics_savepoint()
        )
        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        grg = interpolation_savepoint.geofac_grg()
        solve_nonhydro_interpolation_state = InterpolationState(
            c_lin_e=interpolation_savepoint.c_lin_e(),
            c_intp=interpolation_savepoint.c_intp(),
            e_flx_avg=interpolation_savepoint.e_flx_avg(),
            geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
            geofac_rot=interpolation_savepoint.geofac_rot(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            e_bln_c_s=as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), CEDim),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        solve_nonhydro_metric_state = MetricStateNonHydro(
            bdy_halo_c=metrics_savepoint.bdy_halo_c(),
            mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
            rayleigh_w=metrics_savepoint.rayleigh_w(),
            exner_exfac=metrics_savepoint.exner_exfac(),
            exner_ref_mc=metrics_savepoint.exner_ref_mc(),
            wgtfac_c=metrics_savepoint.wgtfac_c(),
            wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            rho_ref_mc=metrics_savepoint.rho_ref_mc(),
            theta_ref_mc=metrics_savepoint.theta_ref_mc(),
            vwind_expl_wgt=metrics_savepoint.vwind_expl_wgt(),
            d_exner_dz_ref_ic=metrics_savepoint.d_exner_dz_ref_ic(),
            ddqz_z_half=metrics_savepoint.ddqz_z_half(),
            theta_ref_ic=metrics_savepoint.theta_ref_ic(),
            d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
            d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
            rho_ref_me=metrics_savepoint.rho_ref_me(),
            theta_ref_me=metrics_savepoint.theta_ref_me(),
            ddxn_z_full=metrics_savepoint.ddxn_z_full(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
            ipeidx_dsl=metrics_savepoint.ipeidx_dsl(),
            pg_exdist=metrics_savepoint.pg_exdist(),
            ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
            ddxt_z_full=metrics_savepoint.ddxt_z_full(),
            wgtfac_e=metrics_savepoint.wgtfac_e(),
            wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels),
            vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
            hmask_dd3d=metrics_savepoint.hmask_dd3d(),
            scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
            coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
            coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
            coeff_gradekin=metrics_savepoint.coeff_gradekin(),
        )

        diagnostic_metric_state = DiagnosticMetricState(
            ddqz_z_full=metrics_savepoint.ddqz_z_full(),
            rbf_vec_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
            rbf_vec_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
        )

        return (
            diffusion_metric_state,
            diffusion_interpolation_state,
            solve_nonhydro_metric_state,
            solve_nonhydro_interpolation_state,
            diagnostic_metric_state,
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(
    run_path: str, experiment_name: str, processor_procs: ProcessProperties = None
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation

    """
    run_dir = Path(run_path).absolute() if run_path else Path(__file__).absolute().parent
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{experiment_name}.log")
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    # TODO (Chia Rui): modify here when single_dispatch is ready
    console_handler.addFilter(ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
