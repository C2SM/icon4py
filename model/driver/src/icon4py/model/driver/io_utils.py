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

from gt4py.next import Field

from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    InterpolationState,
    MetricStateNonHydro,
    DiagnosticStateNonHydro,
    PrepAdvection,
)
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.decomposition.definitions import DecompositionInfo, ProcessProperties
from icon4py.model.common.decomposition.mpi_decomposition import ParallelLogger
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils import serialbox_utils as sb


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
def model_initialization():
    # create two prognostic states, nnow and nnew?
    # at least two prognostic states are global because they are needed in the dycore, AND possibly nesting and restart processes in the future
    # one is enough for the JW test
    prognostic_state_1 = PrognosticState(
        w=None,
        vn=None,
        theta_v=None,
        rho=None,
        exner=None,
    )
    prognostic_state_2 = PrognosticState(
        w=None,
        vn=None,
        theta_v=None,
        rho=None,
        exner=None,
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
        solve_nonhydro_init_savepoint.divdamp_fac_o2(),
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
