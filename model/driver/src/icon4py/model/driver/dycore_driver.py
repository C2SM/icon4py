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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import click
import pytz
from devtools import Timer
from gt4py.next import Field, program
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.atmosphere.diffusion.diffusion_utils import _identity_c_k, _identity_e_k

from icon4py.model.common.grid.horizontal import EdgeParams, CellParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.states.prognostic_state import PrognosticState, copy_prognostics
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.driver.icon_configuration import IconRunConfig, AtmoNonHydroConfig, read_config
from icon4py.model.driver.io_utils import (
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)

from icon4py.model.common.test_utils.helpers import zero_field

log = logging.getLogger(__name__)


# TODO (magdalena) to be removed once there is a proper time stepping
@program
def _copy_diagnostic_and_prognostics(
    hdef_ic_new: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    div_ic_new: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    dwdx_new: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy_new: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
):
    _identity_c_k(hdef_ic_new, out=hdef_ic)
    _identity_c_k(div_ic_new, out=div_ic)
    _identity_c_k(dwdx_new, out=dwdx)
    _identity_c_k(dwdy_new, out=dwdy)
    _identity_e_k(vn_new, out=vn)
    _identity_c_k(w_new, out=w)
    _identity_c_k(exner_new, out=exner)
    _identity_c_k(theta_v_new, out=theta_v)



class TimeLoop:

    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(self):
        self.run_config: IconRunConfig = None
        self.nonhydro_config: AtmoNonHydroConfig = None
        self.diffusion: Diffusion = None
        self.non_hydro_solver: SolveNonhydro = None

    def init(
        self,
        run_config: IconRunConfig,
        nonhydro_config: AtmoNonHydroConfig,
        diffusion: Diffusion,
        non_hydro_solver: SolveNonhydro
    ):
        self.run_config = run_config
        self.nonhydro_config = nonhydro_config
        self.n_time_steps: int = int((self.run_config.end_date - self.run_config.start_date) / timedelta(seconds=self.run_config.dtime))
        self.diffusion = diffusion
        self.non_hydro_solver = non_hydro_solver

        # check validity of configurations
        self._validate()

        # current simulation date
        #self.simulation_date = datetime.fromisoformat(SIMULATION_START_DATE)
        self._simulation_date: datetime = self.run_config.start_date

    def _validate(self):
        if (self.n_time_steps < 0):
            raise ValueError("end_date should be larger than start_date. Please check.")
        if (not self.diffusion.initialized):
            raise Exception("diffusion is not initialized before time loop")
        if (not self.non_hydro_solver.initialized):
            raise Exception("nonhydro solver is not initialized before time loop")


    def _next_simulation_date(self):
        self._simulation_date += timedelta(seconds=self.run_config.dtime)

    @property
    def simulation_date(self):
        return self._simulation_date

    def _full_name(self, func: Callable):
        return ":".join((self.__class__.__name__, func.__name__))

    def time_integration(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        # below is a long list of arguments for dycore time_step (suggestion: many can be moved to initialization of SolveNonhydro)
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams, # precomputed constants
        z_fields: ZFields, # local constants in solve_nh
        nh_constants: NHConstants, # user defined constants
        cfl_w_limit: float, # need to check
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float], # precomputed constants
        c_owner_mask: Field[[CellDim], bool], # precomputed constants
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float], # in EdgeParams, precomputed constants
        bdy_divdamp: Field[[KDim], float],
        l_recompute: bool,
        l_init: bool,
        lclean_mflx: bool,
        lprep_adv: bool
    ):
        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self.n_time_steps}"
        )
        # TODO remove this l_init? l_init is true when the first time time_integration is called
        if (self.diffusion.config.apply_to_horizontal_wind and l_init and not self.run_config.is_testcase):
            log.info("running initial step to diffuse fields before timeloop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_state,
                self.run_config.dtime,
            )
        log.info(
            f"starting real time loop for dtime={self.run_config.dtime} n_timesteps={self.n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for i_time_step in range(self.n_time_steps):
            log.info(f"run timestep : {i_time_step}")

            # update boundary condition
            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state,
                prep_adv,
                nonhydro_config,
                nonhydro_params,
                edge_geometry,
                z_fields,
                nh_constants,
                cfl_w_limit,
                scalfac_exdiff,
                cell_areas,
                c_owner_mask,
                f_e,
                area_edge,
                bdy_divdamp,
                l_recompute,
                l_init,
                lclean_mflx,
                lprep_adv
            )
            timer.capture()

            # TODO IO

            self._next_simulation_date()
        timer.summary(True)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        c_owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        l_recompute: bool,
        l_init: bool,
        lclean_mflx: bool,
        lprep_adv: bool
    ):
        self._do_dyn_substepping(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prognostic_state,
            prep_adv,
            nonhydro_config,
            nonhydro_params,
            edge_geometry,
            z_fields,
            nh_constants,
            cfl_w_limit,
            scalfac_exdiff,
            cell_areas,
            c_owner_mask,
            f_e,
            area_edge,
            bdy_divdamp,
            l_recompute,
            l_init,
            lclean_mflx,
            lprep_adv
        )

        if (self.diffusion.config.apply_to_horizontal_wind and self.nonhydro_config.apply_horizontal_diff_at_large_dt):
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_state,
                self.run_config.dtime
            )

        # TODO tracer advection

    def _do_dyn_substepping(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        c_owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        l_recompute: bool,
        l_init: bool,
        lclean_mflx: bool,
        lprep_adv: bool
    ):

        # TODO compute airmass for prognostic_state

        rho_new = zero_field(self.non_hydro_solver.grid,CellDim,KDim)
        w_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        vn_new = zero_field(self.non_hydro_solver.grid, EdgeDim, KDim)
        exner_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        theta_v_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        copy_prognostics.with_backend(run_gtfn)(
            prognostic_state.rho,
            rho_new,
            prognostic_state.w,
            w_new,
            prognostic_state.vn,
            vn_new,
            prognostic_state.exner,
            exner_new,
            prognostic_state.theta_v,
            theta_v_new,
        )
        prognostic_state_new = PrognosticState(
            rho=rho_new,
            w=w_new,
            vn=vn_new,
            exner=exner_new,
            theta_v=theta_v_new
        )
        prognostic_state_list = [prognostic_state, prognostic_state_new]
        time_n = 1
        time_n_plus_1 = 0
        for idyn_timestep in range(self.nonhydro_config.ndyn_substeps):
            time_n_swap = time_n_plus_1
            time_n_plus_1 = time_n
            time_n = time_n_swap
            self.non_hydro_solver.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                config=nonhydro_config,
                params=nonhydro_params,
                edge_geometry=edge_geometry,
                z_fields=z_fields,
                nh_constants=nh_constants,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                cell_areas=cell_areas,
                c_owner_mask=c_owner_mask,
                f_e=f_e,
                area_edge=area_edge,
                bdy_divdamp=bdy_divdamp,
                dtime=self.run_config.dtime,
                idyn_timestep=idyn_timestep,
                l_recompute=l_recompute,
                l_init=l_init,
                nnew=time_n_plus_1,
                nnow=time_n,
                lclean_mflx=lclean_mflx,
                lprep_adv=lprep_adv
            )
            if (self.diffusion.config.apply_to_horizontal_wind and not self.nonhydro_config.apply_horizontal_diff_at_large_dt):
                self.diffusion.run(
                    diffusion_diagnostic_state,
                    prognostic_state_list[time_n_plus_1],
                    self.run_config.dtime
                )

        if (time_n_plus_1 == 1):
            copy_prognostics.with_backend(run_gtfn)(
                prognostic_state_new.rho,
                prognostic_state.rho,
                prognostic_state_new.w,
                prognostic_state.w,
                prognostic_state_new.vn,
                prognostic_state.vn,
                prognostic_state_new.exner,
                prognostic_state.exner,
                prognostic_state_new.theta_v,
                prognostic_state.theta_v,
            )

        # TODO compute airmass for prognostic_state

# step_date_init and step_date_exit are used in diffusion, declared in fixtures
@pytest.mark.datatest
@pytest.mark.parametrize(
    "timeloop_date_init, timeloop_date_exit", "linit",
    "istep, step_date_init, step_date_exit",
    [("2021-06-20T12:00:10.000", "2021-06-20T12:00:20.000", True, "2021-06-20T12:00:10.000", "2021-06-20T12:00:20.000")],
)
def test_run_timeloop_single_step(
    timeloop_date_init,
    timeloop_date_exit,
    data_provider,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    interpolation_savepoint,
    r04b09_diffusion_config,
    damping_height,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    savepoint_timeloop_nonhydro_init,
    savepoint_timeloop_velocity_init,
    savepoint_timeloop_velocity_exit,
    savepoint_timeloop_nonhydro_step_exit,
):
    diffusion_config = r04b09_diffusion_config
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    diffusion_interpolation_state = interpolation_savepoint.construct_interpolation_state_for_diffusion()
    diffusion_metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    diffusion_diagnostic_state = diffusion_savepoint_init.construct_diagnostics_for_diffusion()
    before_dycore_prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vct_a = grid_savepoint.vct_a()
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    additional_parameters = DiffusionParams(diffusion_config)

    #verify_diffusion_fields(diffusion_diagnostic_state, before_dycore_prognostic_state, diffusion_savepoint_init)

    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=diffusion_config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    assert diffusion_savepoint_init.fac_bdydiff_v() == diffusion.fac_bdydiff_v

    '''
    verify_diffusion_fields(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    '''

    nonhydro_config = NonHydrostaticConfig()
    sp = savepoint_timeloop_nonhydro_init #savepoint_nonhydro_init
    sp_step_exit = savepoint_timeloop_nonhydro_step_exit
    nonhydro_params = NonHydrostaticParams(nonhydro_config)
    #sp_d = data_provider.from_savepoint_grid()
    sp_v = savepoint_timeloop_velocity_init
    mesh = SimpleMesh()
    dtime = sp_v.get_metadata("dtime").get("dtime")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = PrepAdvection(
        vn_traj=sp.vn_traj(), mass_flx_me=sp.mass_flx_me(), mass_flx_ic=sp.mass_flx_ic()
    )

    enh_smag_fac = zero_field(mesh, KDim)
    a_vec = random_field(mesh, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)
    nnow = 0
    nnew = 1
    recompute = sp_v.get_metadata("recompute").get("recompute")
    linit = sp_v.get_metadata("linit").get("linit")
    dyn_timestep = sp_v.get_metadata("dyn_timestep").get("dyn_timestep")

    diagnostic_state_nh = DiagnosticStateNonHydro(
        theta_v_ic=sp.theta_v_ic(),
        exner_pr=sp.exner_pr(),
        rho_ic=sp.rho_ic(),
        ddt_exner_phy=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_fl_e=sp.mass_fl_e(),
        ddt_vn_phy=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )

    prognostic_state_nnow = PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )

    prognostic_state_nnew = PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )

    z_fields = ZFields(
        z_gradh_exner=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_alpha=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_beta=_allocate(CellDim, KDim, mesh=icon_grid),
        z_w_expl=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_exner_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_q=_allocate(CellDim, KDim, mesh=icon_grid),
        z_contr_w_fl_l=_allocate(CellDim, KDim, is_halfdim=True, mesh=icon_grid),
        z_rho_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_theta_v_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_graddiv_vn=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_rho_expl=_allocate(CellDim, KDim, mesh=icon_grid),
        z_dwdz_dd=_allocate(CellDim, KDim, mesh=icon_grid),
        z_kin_hor_e=_allocate(EdgeDim, KDim, mesh=icon_grid),
        z_vt_ie=_allocate(EdgeDim, KDim, mesh=icon_grid),
    )

    nh_constants = NHConstants(
        wgt_nnow_rth=sp.wgt_nnow_rth(),
        wgt_nnew_rth=sp.wgt_nnew_rth(),
        wgt_nnow_vel=sp.wgt_nnow_vel(),
        wgt_nnew_vel=sp.wgt_nnew_vel(),
        scal_divdamp=sp.scal_divdamp(),
        scal_divdamp_o2=sp.scal_divdamp_o2(),
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_nonhydro()
    metric_state_nonhydro = metrics_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        cell_areas=cell_geometry.area,
        fac=fac,
        z=z,
    )

# TODO initialization of prognostic variables and topography of Jablonowski Williamson test
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

def initialize(file_path: Path, props: ProcessProperties):
    log.info("initialize parallel runtime")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name, n_time_steps=n_time_steps)

    decomp_info = read_decomp_info(file_path, props)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(file_path, rank=props.rank)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        file_path, rank=props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(file_path)

    log.info("initializing diffusion")
    diffusion_params = DiffusionParams(config.diffusion_config)
    exchange = create_exchange(props, decomp_info)
    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    data_provider, diagnostic_state, prognostic_state = read_initial_state(
        file_path, rank=props.rank
    )

    atmo_non_hydro = DummyAtmoNonHydro(data_provider)
    atmo_non_hydro.init(config=config.dycore_config)

    tl = Timeloop(
        config=config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
    )
    return tl, diagnostic_state, prognostic_state

def initialize_original(n_time_steps, file_path: Path, props: ProcessProperties):
    """
    Inititalize the driver run.

    "reads" in
        - configuration

        - grid information

        - (serialized) input fields, initial

     Returns:
         tl: configured timeloop,
         prognostic_state: initial state fro prognostic and diagnostic variables
         diagnostic_state:
    """
    log.info("initialize parallel runtime")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name, n_time_steps=n_time_steps)

    decomp_info = read_decomp_info(file_path, props)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(file_path, rank=props.rank)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        file_path, rank=props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(file_path)

    log.info("initializing diffusion")
    diffusion_params = DiffusionParams(config.diffusion_config)
    exchange = create_exchange(props, decomp_info)
    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    data_provider, diagnostic_state, prognostic_state = read_initial_state(
        file_path, rank=props.rank
    )

    atmo_non_hydro = DummyAtmoNonHydro(data_provider)
    atmo_non_hydro.init(config=config.dycore_config)

    tl = Timeloop(
        config=config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
    )
    return tl, diagnostic_state, prognostic_state

@click.command()
@click.argument("input_path")
@click.option("--run_path", default="", help="folder for output")
@click.option("--n_steps", default=5, help="number of time steps to run, max 5 is supported")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
def main(input_path, run_path, n_steps, mpi):
    """
    Run the driver.

    usage: python driver/dycore_driver.py ../../tests/ser_icondata/mch_ch_r04b09_dsl/ser_data

    steps:
    1. initialize model:

        a) load config

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) setup the time loop

    2. run time loop

    """


    timeloop = TimeLoop()
    timeloop.init()
    #start_time = datetime.now().astimezone(pytz.UTC)
    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    configure_logging(run_path, start_time, parallel_props)
    log.info(f"Starting ICON dycore run: {datetime.isoformat(start_time)}")
    log.info(f"input args: input_path={input_path}, n_time_steps={n_steps}")
    timeloop, diagnostic_state, prognostic_state = initialize(
        n_steps, Path(input_path), parallel_props
    )
    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    timeloop(diagnostic_state, prognostic_state)

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
