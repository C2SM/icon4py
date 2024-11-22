# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import logging
import pathlib
import uuid
from typing import Callable, NamedTuple

import click
from devtools import Timer
from gt4py.next import gtfn_cpu

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import (
    diffusion,
    diffusion_states,
)
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.driver import (
    icon4py_configuration as driver_config,
    initialization_utils as driver_init,
)


log = logging.getLogger(__name__)


class TimeLoop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: driver_config.Icon4pyRunConfig,
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.run_config: driver_config.Icon4pyRunConfig = run_config
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date) / self.run_config.dtime
        )
        self.dtime_in_seconds: float = self.run_config.dtime.total_seconds()
        self._n_substeps_var: int = self.run_config.n_substeps
        self._substep_timestep: float = float(self.dtime_in_seconds / self._n_substeps_var)

        self._validate_config()

        # current simulation date
        self._simulation_date: datetime.datetime = self.run_config.start_date

        self._is_first_step_in_simulation: bool = not self.run_config.restart_mode

    def re_init(self):
        self._simulation_date = self.run_config.start_date
        self._is_first_step_in_simulation = True
        self._n_substeps_var = self.run_config.n_substeps

    def _validate_config(self):
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    @property
    def first_step_in_simulation(self):
        return self._is_first_step_in_simulation

    def _is_last_substep(self, step_nr: int):
        return step_nr == (self.n_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int):
        return step_nr == 0

    def _next_simulation_date(self):
        self._simulation_date += self.run_config.dtime

    @property
    def n_substeps_var(self):
        return self._n_substeps_var

    @property
    def simulation_date(self):
        return self._simulation_date

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @property
    def substep_timestep(self):
        return self._substep_timestep

    def _full_name(self, func: Callable):
        return ":".join((self.__class__.__name__, func.__name__))

    def time_integration(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        # TODO (Chia Rui): expand the PrognosticState to include indices of now and next, now it is always assumed that now = 0, next = 1 at the beginning
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        # below is a long list of arguments for dycore time_step that many can be moved to initialization of SolveNonhydro)
        prep_adv: dycore_states.PrepAdvection,
        initial_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        log.info(
            f"starting time loop for dtime={self.dtime_in_seconds} s and n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self.run_config.apply_initial_stabilization} dtime={self.dtime_in_seconds} s, substep_timestep={self._substep_timestep}"
        )

        # TODO (Chia Rui): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        # TODO (Chia Rui): Compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

        # TODO (Chia Rui): Initialize exner_pr used in solve_nh (compute_exner_pert subroutine)

        if (
            self.diffusion.config.apply_to_horizontal_wind
            and self.run_config.apply_initial_stabilization
            and self._is_first_step_in_simulation
        ):
            log.info("running initial step to diffuse fields before time loop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_states.current,
                self.dtime_in_seconds,
            )
        log.info(
            f"starting real time loop for dtime={self.dtime_in_seconds} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(f"simulation date : {self._simulation_date} run timestep : {time_step}")
            log.info(
                f" MAX VN: {prognostic_states.current.vn.asnumpy().max():.15e} , MAX W: {prognostic_states.current.w.asnumpy().max():.15e}"
            )
            log.info(
                f" MAX RHO: {prognostic_states.current.rho.asnumpy().max():.15e} , MAX THETA_V: {prognostic_states.current.theta_v.asnumpy().max():.15e}"
            )
            # TODO (Chia Rui): check with Anurag about printing of max and min of variables.

            self._next_simulation_date()

            # update boundary condition

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv,
                initial_divdamp_fac_o2,
                do_prep_adv,
            )
            timer.capture()

            # TODO (Chia Rui): modify n_substeps_var if cfl condition is not met. (set_dyn_substeps subroutine)

            # TODO (Chia Rui): compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

            # TODO (Chia Rui): simple IO enough for JW test

        timer.summary(True)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        initial_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): Add update_spinup_damping here to compute divdamp_fac_o2

        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_states,
            prep_adv,
            initial_divdamp_fac_o2,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self.dtime_in_seconds,
            )

        prognostic_states.swap()

        # TODO (Chia Rui): add tracer advection here

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        initial_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): compute airmass for prognostic_state here

        do_recompute = True
        do_clean_mflx = True
        for dyn_substep in range(self._n_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} substep / n_substeps : {dyn_substep} / "
                f"{self.n_substeps_var} , is_first_step_in_simulation : {self._is_first_step_in_simulation}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv=prep_adv,
                divdamp_fac_o2=initial_divdamp_fac_o2,
                dtime=self._substep_timestep,
                l_recompute=do_recompute,
                l_init=self._is_first_step_in_simulation,
                lclean_mflx=do_clean_mflx,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )

            do_recompute = False
            do_clean_mflx = False

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()

            self._is_first_step_in_simulation = False

        # TODO (Chia Rui): compute airmass for prognostic_state here


class DriverStates(NamedTuple):
    """
    Initialized states for the driver run.

    Attributes:
        prep_advection_prognostic: Fields collecting data for advection during the solve nonhydro timestep.
        solve_nonhydro_diagnostic: Initial state for solve_nonhydro diagnostic variables.
        diffusion_diagnostic: Initial state for diffusion diagnostic variables.
        prognostics: Initial state for prognostic variables (double buffered).
        diagnostic: Initial state for global diagnostic variables.
    """

    prep_advection_prognostic: dycore_states.PrepAdvection
    solve_nonhydro_diagnostic: dycore_states.DiagnosticStateNonHydro
    diffusion_diagnostic: diffusion_states.DiffusionDiagnosticState
    prognostics: common_utils.TimeStepPair[prognostics.PrognosticState]
    diagnostic: diagnostics.DiagnosticState


class DriverParams(NamedTuple):
    """
    Parameters for the driver run.

    Attributes:
        divdamp_fac_o2: Second order divergence damping factor.
    """

    divdamp_fac_o2: float


def initialize(
    file_path: pathlib.Path,
    props: decomposition.ProcessProperties,
    serialization_type: driver_init.SerializationType,
    experiment_type: driver_init.ExperimentType,
    grid_id: uuid.UUID,
    grid_root,
    grid_level,
) -> tuple[TimeLoop, DriverStates, DriverParams]:
    """
    Initialize the driver run.

    This function does the following:
    - load configuration
    - load grid information
    - initialize components: diffusion and solve_nh
    - load diagnostic and prognostic variables (serialized data)
    - setup the time loop

    Parameters:
        file_path: Path to the serialized data.
        props: Processor properties.
        serialization_type: Serialization type.
        experiment_type: Experiment type.
        grid_id: Grid ID.
        grid_root: Grid root.
        grid_level: Grid level.

    Returns:
        TimeLoop: Time loop object.
        DriverStates: Initial states for the driver run.
        DriverParams: Parameters for the driver run.
    """
    log.info("initialize parallel runtime")
    log.info(f"reading configuration: experiment {experiment_type}")
    config = driver_config.read_config(experiment_type)

    decomp_info = driver_init.read_decomp_info(
        file_path, props, serialization_type, grid_id, grid_root, grid_level
    )

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = driver_init.read_icon_grid(
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
        grid_id=grid_id,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    log.info(f"reading input fields from '{file_path}'")
    (
        edge_geometry,
        cell_geometry,
        vertical_geometry,
        c_owner_mask,
    ) = driver_init.read_geometry_fields(
        file_path,
        vertical_grid_config=config.vertical_grid_config,
        rank=props.rank,
        ser_type=serialization_type,
        grid_id=grid_id,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    ) = driver_init.read_static_fields(
        icon_grid,
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
    )

    log.info("initializing diffusion")
    diffusion_params = diffusion.DiffusionParams(config.diffusion_config)
    exchange = decomposition.create_exchange(props, decomp_info)
    diffusion_granule = diffusion.Diffusion(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        diffusion_metric_state,
        diffusion_interpolation_state,
        edge_geometry,
        cell_geometry,
        exchange=exchange,
        backend=gtfn_cpu,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config.solve_nonhydro_config)

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        backend=gtfn_cpu,
        config=config.solve_nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_geometry,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=c_owner_mask,
    )

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        initial_divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = driver_init.read_initial_state(
        icon_grid,
        cell_geometry,
        edge_geometry,
        file_path,
        rank=props.rank,
        experiment_type=experiment_type,
    )
    prognostics_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    time_loop = TimeLoop(
        run_config=config.run_config,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )

    return (
        time_loop,
        DriverStates(
            prep_advection_prognostic=prep_adv,
            solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
            diffusion_diagnostic=diffusion_diagnostic_state,
            prognostics=prognostics_states,
            diagnostic=diagnostic_state,
        ),
        DriverParams(divdamp_fac_o2=initial_divdamp_fac_o2),
    )


@click.command()
@click.argument("input_path")
@click.option(
    "--run_path",
    default="./",
    help="Folder for code logs to file and to console. Only debug logging is going to the file.",
)
@click.option(
    "--mpi",
    default=False,
    show_default=True,
    help="Whether or not you are running with mpi. Currently not fully tested yet.",
)
@click.option(
    "--serialization_type",
    default="serialbox",
    show_default=True,
    help="Serialization type for grid info and static fields. This is currently the only possible way to load the grid info and static fields.",
)
@click.option(
    "--experiment_type",
    default="any",
    show_default=True,
    help="Option for configuration and how the initial state is generated. "
    "Setting it to the default value will instruct the model to use the default configuration of MeteoSwiss regional experiment and read the initial state from serialized data. "
    "Currently, users can also set it to either jabw or grauss_3d_torus to generate analytic initial condition for the JW and mountain wave tests, respectively (they are placed in abs_path_to_icon4py/model/driver/src/icon4py/model/driver/test_cases/).",
)
@click.option(
    "--grid_root",
    default=2,
    show_default=True,
    help="Grid root division (please refer to Sadourny et al. 1968 or ICON documentation for more information). When torus grid is used, it must be set to 2.",
)
@click.option(
    "--grid_level",
    default=4,
    show_default=True,
    help="Grid refinement level. When torus grid is used, it must be set to 0.",
)
@click.option(
    "--grid_id",
    default="af122aca-1dd2-11b2-a7f8-c7bf6bc21eba",
    help="uuid of the horizontal grid ('uuidOfHGrid' from gridfile)",
)
def icon4py_driver(
    input_path, run_path, mpi, serialization_type, experiment_type, grid_id, grid_root, grid_level
) -> None:
    """
    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data

    Run the icon4py driver, where INPUT_PATH is the path to folder storing the serialized data.

    steps:
    1. initialize model from serialized data:

        a) load config of icon and components: diffusion and solve_nh

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) load local, diagnostic and prognostic variables

        e) setup the time loop

    2. run time loop
    """
    parallel_props = decomposition.get_processor_properties(decomposition.get_runtype(with_mpi=mpi))
    grid_id = uuid.UUID(grid_id)
    driver_init.configure_logging(run_path, experiment_type, parallel_props)

    time_loop: TimeLoop
    ds: DriverStates
    dp: DriverParams
    time_loop, ds, dp = initialize(
        pathlib.Path(input_path),
        parallel_props,
        serialization_type,
        experiment_type,
        grid_id,
        grid_root,
        grid_level,
    )
    log.info(f"Starting ICON dycore run: {time_loop.simulation_date.isoformat()}")
    log.info(
        f"input args: input_path={input_path}, n_time_steps={time_loop.n_time_steps}, ending date={time_loop.run_config.end_date}"
    )

    log.info(f"input args: input_path={input_path}, n_time_steps={time_loop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("time loop: START")

    time_loop.time_integration(
        ds.diffusion_diagnostic,
        ds.solve_nonhydro_diagnostic,
        ds.prognostics,
        ds.prep_advection_prognostic,
        dp.divdamp_fac_o2,
        do_prep_adv=False,
    )

    log.info("time loop:  DONE")


if __name__ == "__main__":
    icon4py_driver()
