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
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from gt4py.next import Field

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.interpolation_state import InterpolationState
from icon4py.model.atmosphere.dycore.state_utils.metric_state import MetricStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate, zero_field
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.decomposition.definitions import DecompositionInfo, ProcessProperties
from icon4py.model.common.decomposition.mpi_decomposition import ParallelLogger
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.common.constants import GRAV, RD, EARTH_RADIUS, EARTH_ANGULAR_VELOCITY, MATH_PI, MATH_PI_2, RD_O_CPD, P0REF, CVD_O_RD

from gt4py.next import as_field

from icon4py.model.common.interpolation.stencils import mo_cells2edges_scalar_interior

SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, Enum):
    SB = "serialbox"
    NC = "netcdf"


def read_icon_grid(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads
        from ppser serialized test data
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid()
            .construct_icon_grid()
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


# TODO (Chia Rui): initialization of prognostic variables and topography of Jablonowski Williamson test
def model_initialization(
    cell_param: CellParams,
    edge_param: EdgeParams,
    interpolation_savepoint,
    geopot: Field[[CellDim,KDim], float],
    num_levels: int,
    p_sfc: float,
    jw_up: float,
    jw_u0: float,
    jw_temp0: float,
    path: Path,
    rank=0
):
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )
    icon_grid = (
        sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
        .from_savepoint_grid()
        .construct_icon_grid()
    )
    cells2edges_coeff = data_provider.from_interpolation_savepoint().c_lin_e()

    #jw_up = 1.0
    #jw_u0 = 35.0
    #jw_temp0 = 288.
    # DEFINED PARAMETERS for jablonowski williamson:
    eta_0 = 0.252
    eta_t = 0.2    # tropopause
    gamma = 0.005 # temperature elapse rate (K/m)
    dtemp = 4.8e5 # empirical temperature difference (K)
    # for baroclinic wave test in the future
    lonC = MATH_PI / 9.0 # longitude of the perturb centre
    latC = 2.0 * lonC # latitude of the perturb centre
    ps_o_p0ref = p_sfc / P0REF
    cpd_o_rd = 1.0 / RD_O_CPD

    # create two prognostic states, nnow and nnew?
    # at least two prognostic states are global because they are needed in the dycore, AND possibly nesting and restart processes in the future
    # one is enough for the JW test
    cell_lat = np.asarray(cell_param.center_lat)
    cell_lon = np.asarray(cell_param.center_lon)
    edge_lat = np.asarray(edge_param.center[0])
    edge_lon = np.asarray(edge_param.center[1])
    primal_normal_x = np.asarray(edge_param.primal_normal[0])
    primal_normal_y = np.asarray(edge_param.primal_normal[1])
    geopot_numpy = np.asarray(geopot)
    cell_size = cell_lat.size
    edge_size = edge_lat.size
    vn_numpy = np.zeros((edge_size, num_levels), dtype=float)
    w_numpy = np.zeros((cell_size, num_levels+1), dtype=float)
    exner_numpy = np.zeros((cell_size, num_levels), dtype=float)
    rho_numpy = np.zeros((cell_size, num_levels), dtype=float)
    temperature_numpy = np.zeros((cell_size, num_levels), dtype=float)
    pressure_numpy = np.zeros((cell_size, num_levels), dtype=float)
    theta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)

    sin_lat = np.sin(cell_lat)
    cos_lat = np.cos(cell_lat)
    fac1 = 1.0 / 6.3 - 2.0 * (sin_lat ** 6) * (cos_lat ** 2 + 1.0/ 3.0)
    fac2 = (8.0/ 5.0 * (cos_lat ** 3) * (sin_lat ** 2 + 2.0/ 3.0) - 0.25 * MATH_PI) * EARTH_RADIUS * EARTH_ANGULAR_VELOCITY
    lapse_rate = RD * gamma / GRAV
    for k_index in range(num_levels-1,0,-1):
        eta_old = np.full(cell_size,fill_value=1.0e-7, dtype=float)
        # Newton iteration to determine zeta
        for newton_iter_index in range(100):
            eta_v = (eta_old - eta_0) * MATH_PI_2
            cos_etav = np.cos(eta_v)
            sin_etav = np.sin(eta_v)

            temperature_avg = jw_temp0 * (eta_old ** lapse_rate)
            geopot_avg = jw_temp0 * GRAV / gamma * (1.0 - eta_old ** lapse_rate)
            temperature_avg = np.where(
                eta_old < eta_t,
                temperature_avg + dtemp * ((eta_t - eta_old) ** 5),
                temperature_avg
            )
            geopot_avg = np.where(
                eta_old < eta_t,
                geopot_avg - RD * dtemp * ((np.log(eta_old / eta_t) + 137.0 / 60.0) * (eta_t ** 5) - 5.0 * (eta_t ** 4) * eta_old + 5.0 * (eta_t ** 3) * (eta_old ** 2) - 10.0 / 3.0 * (eta_t ** 2) * (eta_old ** 3) + 1.25 * eta_t * (eta_old ** 4) - 0.2 * (eta_old ** 5)),
                geopot_avg
            )

            geopot_jw = geopot_avg + jw_u0 * (cos_etav ** 1.5) * (fac1 * jw_u0 * (cos_etav ** 1.5) + fac2)
            temperature_jw = temperature_avg + 0.75 * eta_old * MATH_PI * jw_u0 / RD * sin_etav * np.sqrt(cos_etav) * (2.0 * jw_u0 * fac1 * (cos_etav ** 1.5) + fac2)
            newton_function = geopot_jw - geopot_numpy[:, k_index]
            newton_function_prime = -RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v = (eta_old - eta_0) * MATH_PI_2
        # Use analytic expressions at all model level
        exner_numpy[:, k_index] = eta_old * ps_o_p0ref ** RD_O_CPD
        theta_v_numpy[:, k_index] = temperature_jw / exner_numpy[:, k_index]
        rho_numpy[:, k_index] = exner_numpy[:, k_index] ** CVD_O_RD * P0REF / RD / theta_v_numpy[:, k_index]
        #initialize diagnose pressure and temperature variables
        pressure_numpy[:, k_index] = P0REF * exner_numpy[:, k_index] ** cpd_o_rd
        temperature_numpy[:, k_index] = temperature_jw

    eta_v_e = zero_field(icon_grid, EdgeDim, KDim)
    mo_cells2edges_scalar_interior(
        cells2edges_coeff,
        eta_v,
        eta_v_e,

    )

    for edge_index in range(edge_size):
        for k_index in range(num_levels):
            z_lat = edge_lat[edge_index, k_index]
            z_lon = edge_lon[edge_index, k_index]
            zu = jw_u0 * (np.cos(eta_v_e) ** 1.5) * (np.sin(2.0 * z_lat) ** 2)
            if (jw_up > 1.e-20):
                z_fac1 = np.sin(latC) * np.sin(z_lat) + np.cos(latC) * np.cos(z_lat) * np.cos(z_lon - lonC)
                z_fac2 = 10.0 * np.arccos(z_fac1)
                zu = zu + jw_up * np.exp(-z_fac2 ** 2)

            zv = 0.0
            vn_numpy[edge_index, k_index] = zu * primal_normal_x[edge_index, k_index] + zv * primal_normal_y[edge_index, k_index]

    vn = as_field((EdgeDim, KDim), vn_numpy)
    w = as_field((CellDim, KDim), w_numpy)
    exner = as_field((CellDim, KDim), exner_numpy)
    rho = as_field((CellDim, KDim), rho_numpy)
    temperature = as_field((CellDim, KDim), temperature_numpy)
    pressure = as_field((CellDim, KDim), pressure_numpy)
    theta_v = as_field((CellDim, KDim), theta_v_numpy)

    # set surface pressure to the prescribed value
    pres_sfc = as_field((CellDim, KDim), np.full((cell_size, num_levels), fill_value=p_sfc, dtype=float))

    #TODO (Chia Rui): set up diagnostic values of surface pressure, pressure, and temperature

    prognostic_state_1 = PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_2 = PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    return (prognostic_state_1, prognostic_state_2)


def read_initial_state(
    path: Path, rank=0
) -> tuple[
    DiffusionDiagnosticState,
    DiagnosticStateNonHydro,
    ZFields,
    NHConstants,
    PrepAdvection,
    Field[[KDim], float],
    PrognosticState,
    PrognosticState,
]:
    """
    Read prognostic and diagnostic state from serialized data.

    Args:
        path: path to the serialized input data
        rank: mpi rank of the current compute node

    Returns: a tuple containing the data_provider, the initial diagnostic and prognostic state.
        The data_provider is returned such that further timesteps of diagnostics and prognostics
        can be read from within the dummy timeloop

    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )
    icon_grid = (
        sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
        .from_savepoint_grid()
        .construct_icon_grid()
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
    diffusion_diagnostic_state = diffusion_init_savepoint.construct_diagnostics_for_diffusion()
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
    )

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_beta=_allocate(CellDim, KDim, grid=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_q=_allocate(CellDim, KDim, grid=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, grid=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, grid=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, grid=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, grid=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, grid=icon_grid),
    )

    nh_constants = NHConstants(
        wgt_nnow_rth=solve_nonhydro_init_savepoint.wgt_nnow_rth(),
        wgt_nnew_rth=solve_nonhydro_init_savepoint.wgt_nnew_rth(),
        wgt_nnow_vel=solve_nonhydro_init_savepoint.wgt_nnow_vel(),
        wgt_nnew_vel=solve_nonhydro_init_savepoint.wgt_nnew_vel(),
        scal_divdamp=solve_nonhydro_init_savepoint.scal_divdamp(),
        scal_divdamp_o2=solve_nonhydro_init_savepoint.scal_divdamp_o2(),
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
        z_fields,
        nh_constants,
        prep_adv,
        solve_nonhydro_init_savepoint.bdy_divdamp(),
        prognostic_state_now,
        prognostic_state_next,
    )


def read_geometry_fields(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> tuple[EdgeParams, CellParams, VerticalModelParams, Field[[CellDim], bool]]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        ).from_savepoint_grid()
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vertical_geometry = VerticalModelParams(
            vct_a=sp.vct_a(),
            rayleigh_damping_height=12500,
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
) -> DecompositionInfo:
    if ser_type == SerializationType.SB:
        sp = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), True, procs_props.rank
        )
        return sp.from_savepoint_grid().construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    path: Path, rank=0, ser_type: SerializationType = SerializationType.SB
) -> tuple[
    DiffusionMetricState, DiffusionInterpolationState, MetricStateNonHydro, InterpolationState
]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        dataprovider = sb.IconSerialDataProvider(
            "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
        )
        icon_grid = (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid()
            .construct_icon_grid()
        )
        diffusion_interpolation_state = (
            dataprovider.from_interpolation_savepoint().construct_interpolation_state_for_diffusion()
        )
        diffusion_metric_state = (
            dataprovider.from_metrics_savepoint().construct_metric_state_for_diffusion()
        )
        solve_nonhydro_interpolation_state = (
            dataprovider.from_interpolation_savepoint().construct_interpolation_state_for_nonhydro()
        )
        solve_nonhydro_metric_state = (
            dataprovider.from_metrics_savepoint().construct_nh_metric_state(icon_grid.num_levels)
        )
        return (
            diffusion_metric_state,
            diffusion_interpolation_state,
            solve_nonhydro_metric_state,
            solve_nonhydro_interpolation_state,
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(
    run_path: str, start_time: datetime, processor_procs: ProcessProperties = None
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        start_time: start time of the model run

    """
    run_dir = Path(run_path).absolute() if run_path else Path(__file__).absolute().parent
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{datetime.isoformat(start_time)}.log")
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
