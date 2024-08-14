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

import enum
import functools
import logging
import pathlib

from icon4py.model.atmosphere.diffusion import diffusion_states as diffus_states
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.decomposition import (
    definitions as decomposition,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.dimension import (
    CEDim,
    CellDim,
    KDim,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
    serialbox_utils as sb,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.driver import (
    serialbox_helpers as driver_sb,
)
from icon4py.model.driver.test_cases import gauss3d, jablonowski_williamson


GRID_LEVEL = 4
GRID_ROOT = 2
GLOBAL_GRID_ID = dt_utils.GRID_IDS[dt_utils.GLOBAL_EXPERIMENT]

SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."
INITIALIZATION_ERROR_MSG = "The requested experiment type is not implemented."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, enum.Enum):
    SB = "serialbox"
    NC = "netcdf"


class ExperimentType(str, enum.Enum):
    JABW = "jabw"
    """initial condition of Jablonowski-Williamson test"""
    GAUSS3D = "gauss3d_torus"
    """initial condition of Gauss 3D test"""
    ANY = "any"
    """any test with initial conditions read from serialized data (remember to set correct SIMULATION_START_DATE)"""


def read_icon_grid(
    path: pathlib.Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> icon_grid.IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        rank: mpi rank of the current compute node
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads from ppser serialized test data
        grid_id: id (uuid) of the horizontal grid
        grid_root: global grid root division number
        grid_level: global grid refinement number
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid(grid_id, grid_root, grid_level)
            .construct_icon_grid(on_gpu=False)
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def model_initialization_serialbox(
    grid: icon_grid.IconGrid, path: pathlib.Path, rank=0
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
    Initial condition read from serialized data. Diagnostic variables are allocated as zero
    fields.

    Args:
        grid: IconGrid
        path: path where to find the input data
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    data_provider = _serial_data_provider(path, rank)
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
    diffusion_diagnostic_state = driver_sb.construct_diagnostics_for_diffusion(
        diffusion_init_savepoint,
    )
    solve_nonhydro_diagnostic_state = solve_nh_states.DiagnosticStateNonHydro(
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
        exner_dyn_incr=solve_nonhydro_init_savepoint.exner_dyn_incr(),
    )

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        pressure_ifc=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        temperature=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        u=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        v=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )

    prognostic_state_next = prognostics.PrognosticState(
        w=solve_nonhydro_init_savepoint.w_new(),
        vn=solve_nonhydro_init_savepoint.vn_new(),
        theta_v=solve_nonhydro_init_savepoint.theta_v_new(),
        rho=solve_nonhydro_init_savepoint.rho_new(),
        exner=solve_nonhydro_init_savepoint.exner_new(),
    )

    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=solve_nonhydro_init_savepoint.vn_traj(),
        mass_flx_me=solve_nonhydro_init_savepoint.mass_flx_me(),
        mass_flx_ic=solve_nonhydro_init_savepoint.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
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
    grid: icon_grid.IconGrid,
    cell_param: h_grid.CellParams,
    edge_param: h_grid.EdgeParams,
    path: pathlib.Path,
    rank=0,
    experiment_type: ExperimentType = ExperimentType.ANY,
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
    Read initial prognostic and diagnostic fields.

    Args:
        grid: IconGrid
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
        ) = jablonowski_williamson.model_initialization_jabw(
            grid, cell_param, edge_param, path, rank
        )
    elif experiment_type == ExperimentType.GAUSS3D:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = gauss3d.model_initialization_gauss3d(grid, edge_param, path, rank)
    elif experiment_type == ExperimentType.ANY:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_serialbox(grid, path, rank)
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
    path: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> tuple[h_grid.EdgeParams, h_grid.CellParams, v_grid.VerticalGridParams, fa.CellField[bool]]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        vertical_grid_config: Vertical grid configuration
        rank: mpi rank of the current compute node
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read
        grid_id: id (uuid) of the horizontal grid
        grid_root: global grid root division number
        grid_level: global grid refinement number

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = _grid_savepoint(path, rank, grid_id, grid_root, grid_level)
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config)
        vertical_geometry = v_grid.VerticalGridParams(
            vertical_config=vertical_grid_config,
            vct_a=vct_a,
            vct_b=vct_b,
            _min_index_flat_horizontal_grad_pressure=sp.nflat_gradp(),
        )
        return edge_geometry, cell_geometry, vertical_geometry, sp.c_owner_mask()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


@functools.cache
def _serial_data_provider(path, rank) -> sb.IconSerialDataProvider:
    return sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)


@functools.cache
def _grid_savepoint(path, rank, grid_id, grid_root, grid_level) -> sb.IconGridSavepoint:
    sp = _serial_data_provider(path, rank).from_savepoint_grid(grid_id, grid_root, grid_level)
    return sp


def read_decomp_info(
    path: pathlib.Path,
    procs_props: decomposition.ProcessProperties,
    ser_type=SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> decomposition.DecompositionInfo:
    if ser_type == SerializationType.SB:
        return _grid_savepoint(
            path, procs_props.rank, grid_id, grid_root, grid_level
        ).construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    grid: icon_grid.IconGrid,
    path: pathlib.Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
) -> tuple[
    diffus_states.DiffusionMetricState,
    diffus_states.DiffusionInterpolationState,
    solve_nh_states.MetricStateNonHydro,
    solve_nh_states.InterpolationState,
    diagnostics.DiagnosticMetricState,
]:
    """
    Read fields for metric and interpolation state.

     Args:
        grid: IconGrid
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        data_provider = _serial_data_provider(path, rank)

        diffusion_interpolation_state = driver_sb.construct_interpolation_state_for_diffusion(
            data_provider.from_interpolation_savepoint()
        )
        diffusion_metric_state = driver_sb.construct_metric_state_for_diffusion(
            data_provider.from_metrics_savepoint()
        )
        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        grg = interpolation_savepoint.geofac_grg()
        solve_nonhydro_interpolation_state = solve_nh_states.InterpolationState(
            c_lin_e=interpolation_savepoint.c_lin_e(),
            c_intp=interpolation_savepoint.c_intp(),
            e_flx_avg=interpolation_savepoint.e_flx_avg(),
            geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
            geofac_rot=interpolation_savepoint.geofac_rot(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), CEDim),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        solve_nonhydro_metric_state = solve_nh_states.MetricStateNonHydro(
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
            wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(grid.num_levels),
            vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
            hmask_dd3d=metrics_savepoint.hmask_dd3d(),
            scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
            coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
            coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
            coeff_gradekin=metrics_savepoint.coeff_gradekin(),
        )

        diagnostic_metric_state = diagnostics.DiagnosticMetricState(
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
    run_path: str, experiment_name: str, processor_procs: decomposition.ProcessProperties = None
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation

    """
    run_dir = (
        pathlib.Path(run_path).absolute() if run_path else pathlib.Path(__file__).absolute().parent
    )
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
    console_handler.addFilter(mpi_decomp.ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
