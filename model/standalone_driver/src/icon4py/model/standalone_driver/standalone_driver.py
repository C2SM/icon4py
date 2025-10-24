# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import logging
import os
import pathlib
from collections.abc import Callable
from typing import Annotated, NamedTuple

import typer
from devtools import Timer
from gt4py.next import config as gtx_config, metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.standalone_driver import (
    driver_configuration as driver_configure,
    initialization_utils as driver_init,
)


log = logging.getLogger(__name__)


class _DriverFormatter(logging.Formatter):
    def __init__(self, style: str, default_fmt: str, debug_fmt: str):
        super().__init__(fmt=default_fmt, style=style)
        self._debug_formatter = logging.Formatter(fmt=debug_fmt, style=style)

    def format(self, record: logging.LogRecord) -> logging.Formatter:
        if record.levelno == logging.DEBUG:
            return self._debug_formatter.format(record)
        else:
            return super().format(record)


class Driver:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        config: driver_configure.DriverConfig,
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.config: driver_configure.DriverConfig = config
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule
        self._xp = data_alloc.import_array_ns(self.config.backend)
        self._log = logging.getLogger(self.name)

        self._initialize_timeloop_parameters()
        self._validate_config()
        self._filter_log()

    def _filter_log(self):
        file_handler = logging.FileHandler(
            filename=f"log_{self.config.experiment_name}_statistics_{datetime.now(datetime.timezone.utc)}.txt"
        )
        default_log_format = "{message}"
        debug_log_format = "{asctime} {funcName:<20} : {message}"
        file_handler.setFormatter(
            _DriverFormatter(style="{", default_fmt=default_log_format, debug_fmt=debug_log_format)
        )
        file_handler.setLevel(logging.DEBUG)
        self._log.addHandler(file_handler)

    def _initialize_timeloop_parameters(self):
        """
        Initialize parameters needed for running the time loop from start to end date
        """
        self._n_time_steps: int = int(
            (self.config.end_date - self.config.start_date) / self.config.dtime
        )
        self._dtime_in_seconds: ta.wpfloat = self.config.dtime.total_seconds()
        self._ndyn_substeps_var: int = self._update_ndyn_substeps(self.config.ndyn_substeps)
        self._max_ndyn_substeps: int = self.config.ndyn_substeps + 7
        self._substep_timestep: ta.wpfloat = ta.wpfloat(
            self._dtime_in_seconds / self._ndyn_substeps_var
        )
        self._elapse_time_in_seconds: ta.wpfloat = ta.wpfloat("0.0")

        # current simulation date
        self._simulation_date: datetime.datetime = self.config.start_date

        self._is_first_step_in_simulation: bool = True

        self._cfl_watch_mode = False

    def _validate_config(self):
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    def _update_ndyn_substeps(self, new_ndyn_substeps: int):
        self._ndyn_substeps_var = new_ndyn_substeps

    def re_initialization(self):
        self._initialize_timeloop_parameters()
        self._validate_config()

    @property
    def first_step_in_simulation(self):
        return self._is_first_step_in_simulation

    def _is_last_substep(self, step_nr: int):
        return step_nr == (self._ndyn_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int):
        return step_nr == 0

    def _next_simulation_date(self):
        self._simulation_date += self.config.dtime
        self._elapse_time_in_seconds += self.config.dtime

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @property
    def substep_timestep(self):
        return self._substep_timestep

    def _full_name(self, func: Callable):
        return f"{self.__class__.__name__}:{func.__name__}"

    def time_integration(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
        profiling: driver_configure.ProfilingConfig | None = None,
    ):
        log.info(
            f"starting time loop for dtime={self._dtime_in_seconds} s and n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind}, dtime={self._dtime_in_seconds} s, substep_timestep={self._substep_timestep}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        # TODO(OngChia): Compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

        timer_first_timestep = Timer("Driver: first time step", dp=6)
        timer_after_first_timestep = Timer("Driver: after first time step", dp=6)
        for time_step in range(self._n_time_steps):
            timer = timer_first_timestep if time_step == 0 else timer_after_first_timestep
            if profiling is not None:
                if not profiling.skip_first_timestep or time_step > 0:
                    gtx_config.COLLECT_METRICS_LEVEL = profiling.gt4py_metrics_level

            log.info(f"simulation date : {self._simulation_date}, run timestep : {time_step}")
            if self.config.output_statistics:
                rho_arg_max = self._xp.abs(prognostic_states.current.rho.ndarray).max()
                vn_arg_max = self._xp.abs(prognostic_states.current.vn.ndarray).max()
                w_arg_max = self._xp.abs(prognostic_states.current.w.ndarray).max()
                log.info(
                    f" MAX RHO, VN, and W: {prognostic_states.current.rho.ndarray[rho_arg_max]:.5e} / {prognostic_states.current.vn.ndarray[vn_arg_max]:.5e} / {prognostic_states.current.w.ndarray[w_arg_max]:.5e}, at levels: {self._xp.unravel_index(rho_arg_max, prognostic_states.current.rho.ndarray.shape)[1]} / {self._xp.unravel_index(vn_arg_max, prognostic_states.current.vn.ndarray.shape)[1]} / {self._xp.unravel_index(w_arg_max, prognostic_states.current.w.ndarray.shape)[1]}"
                )

            self._next_simulation_date()

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv,
                do_prep_adv,
            )
            device_utils.sync(self.config.backend)
            timer.capture()

            self._is_first_step_in_simulation = False

            # TODO(OngChia): compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

            # TODO(OngChia): simple IO enough for JW test

        timer_first_timestep.summary(True)
        if self.n_time_steps > 1:  # in case only one time step was run
            timer_after_first_timestep.summary(True)
        if profiling is not None and profiling.gt4py_metrics_level > gtx_metrics.DISABLED:
            print(gtx_metrics.dumps())
            gtx_metrics.dump_json(profiling.gt4py_metrics_output_file)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ):
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_states,
            prep_adv,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self._dtime_in_seconds,
            )

        prognostic_states.swap()

    # TODO(OngChia): add tracer advection here

    def _update_time_levels_for_velocity_tendencies(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        at_first_substep: bool,
        at_initial_timestep: bool,
    ):
        """
        Set time levels of advective tendency fields for call to velocity_tendencies.

        When using `TimeSteppingScheme.MOST_EFFICIENT` (itime_scheme=4 in ICON Fortran),
        `vertical_wind_advective_tendency.predictor` (advection term in vertical momentum equation in
        predictor step) is not computed in the predictor step of each substep.
        Instead, the advection term computed in the corrector step during the
        previous substep is reused for efficiency (except, of course, in the
        very first substep of the initial time step).
        `normal_wind_advective_tendency.predictor` (advection term in horizontal momentum equation in
        predictor step) is only computed in the predictor step of the first
        substep and the advection term in the corrector step during the previous
        substep is reused for `normal_wind_advective_tendency.predictor` from the second substep onwards.
        Additionally, in this scheme the predictor and corrector outputs are kept
        in separate elements of the pair (.predictor for the predictor step and
        .corrector for the corrector step) and interpoolated at the end of the
        corrector step to get the final output.

        No other time stepping schemes are currently supported.

        Args:
            diagnostic_state_nh: Diagnostic fields calculated in the dynamical core (SolveNonHydro)
            at_first_substep: Flag indicating if this is the first substep of the time step.
            at_initial_timestep: Flag indicating if this is the first time step.

        Returns:
            The index of the pair element to be used for the corrector output.
        """
        if not (at_initial_timestep and at_first_substep):
            diagnostic_state_nh.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic_state_nh.normal_wind_advective_tendency.swap()

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ):
        # TODO(OngChia): compute airmass for prognostic_state here

        for dyn_substep in range(self._ndyn_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} substep / n_substeps : {dyn_substep} / "
                f"{self._ndyn_substeps_var} , is_first_step_in_simulation : {self._is_first_step_in_simulation}"
            )

            self._update_time_levels_for_velocity_tendencies(
                solve_nonhydro_diagnostic_state,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_initial_timestep=self._is_first_step_in_simulation,
            )

            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv=prep_adv,
                second_order_divdamp_factor=ta.wpfloat("0.0"),
                dtime=self._substep_timestep,
                ndyn_substeps_var=self._ndyn_substeps_var,
                at_initial_timestep=self._is_first_step_in_simulation,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()

        # TODO(OngChia): compute airmass for prognostic_state here

    # watch_mode is true if step is <= 1 or cfl already near or exceeding threshold.
    # omit spinup feature and the option that if the model starts from IFS or COSMO data
    # horizontal cfl is not ported
    def _adjust_ndyn_substeps_var(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
    ):
        skip_checking_watch_mode_quit = False
        # TODO (Chia Rui): perform a global max operation in multinode run
        global_max_vertical_cfl = solve_nonhydro_diagnostic_state.max_vertical_cfl

        if (
            global_max_vertical_cfl > ta.wpfloat("0.81") * self.config.vertical_cfl_threshold
            and not self._cfl_watch_mode
        ):
            log.critical(
                "High CFL number for vertical advection in dynamical core, entering watch mode"
            )
            self._cfl_watch_mode = True

        if self._cfl_watch_mode:
            substep_fraction = ta.wpfloat(self._ndyn_substeps_var / self.config.ndyn_substeps)
            if (
                global_max_vertical_cfl * substep_fraction
                > ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            ):
                log.critical(
                    f"Maximum vertical CFL number {global_max_vertical_cfl} is close to critical threshold"
                )

            vertical_cfl_threshold_for_increment = self.config.vertical_cfl_threshold
            vertical_cfl_threshold_for_decrement = (
                ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            )

            if global_max_vertical_cfl > vertical_cfl_threshold_for_increment:
                ndyn_substeps_increment = max(
                    1,
                    round(
                        self._ndyn_substeps_var
                        * (global_max_vertical_cfl - vertical_cfl_threshold_for_increment)
                        / vertical_cfl_threshold_for_increment
                    ),
                )
                new_ndyn_substeps_var = min(
                    self._ndyn_substeps_var + ndyn_substeps_increment, self._max_ndyn_substeps
                )
                self._update_ndyn_substeps(new_ndyn_substeps_var)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.critical(
                    f"The number of dynamics substeps is increased to {self._ndyn_substeps_var}"
                )
            if (
                self._ndyn_substeps_var > self.config.ndyn_substeps
                and global_max_vertical_cfl
                * ta.wpfloat(self._ndyn_substeps_var / (self._ndyn_substeps_var - 1))
                < vertical_cfl_threshold_for_decrement
            ):
                self._update_ndyn_substeps(self._ndyn_substeps_var - 1)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.critical(
                    f"The number of dynamics substeps is decreaed to {self._ndyn_substeps_var}"
                )
                skip_checking_watch_mode_quit = True
        if (
            not skip_checking_watch_mode_quit
            and self._ndyn_substeps_var == self.config.ndyn_substeps
            and global_max_vertical_cfl < ta.wpfloat("0.76") * self.config.vertical_cfl_threshold
        ):
            log.critical(
                "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
            )
            self._cfl_watch_mode = False

        # reset max_vertical_cfl to zero
        solve_nonhydro_diagnostic_state.max_vertical_cfl = ta.wpfloat("0.0")

    def _update_spinup_second_order_divergence_damping(
        self, fourth_order_divdamp_factor: ta.wpfloat
    ) -> ta.wpfloat:
        if self.config.apply_initial_stabilization:
            if self._elapse_time_in_seconds <= ta.wpfloat("1800.0"):
                return ta.wpfloat("0.8") * fourth_order_divdamp_factor
            elif self._elapse_time_in_seconds <= ta.wpfloat("7200.0"):
                return (
                    ta.wpfloat("0.8")
                    * fourth_order_divdamp_factor
                    * (self._elapse_time_in_seconds - ta.wpfloat("1800.0"))
                    / ta.wpfloat("5400.0")
                )
            else:
                return ta.wpfloat("0.0")
        else:
            return ta.wpfloat("0.0")


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
        second_order_divdamp_factor: Second order divergence damping factor.
    """

    second_order_divdamp_factor: float


def initialize(
    run_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    log_level: str,
    backend: str,
) -> tuple[Driver, DriverStates, DriverParams]:
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
        grid_file: Path of the grid.
        backend: GT4Py backend.

    Returns:
        Driver: Driver object.
        DriverStates: Initial states for the driver run.
        DriverParams: Parameters for the driver run.
    """
    parallel_props = decomposition.get_processor_properties(
        decomposition.get_runtype(with_mpi=False)
    )
    # TODO (Chia Rui): experiment name should be in driver config
    driver_init.configure_logging(run_path, log_level, "Jablownoski-Williamson", parallel_props)

    log.info("initialize parallel runtime")
    log.info("reading configuration: experiment Jablownoski-Williamson")
    driver_config, vertical_config, diffusion_config, solve_nh_config = (
        driver_configure.read_config(backend=backend)
    )

    decomp_info = driver_init.read_decomp_info(
        path=file_path,
        grid_file=grid_file,
        procs_props=props,
        backend=backend,
        ser_type=serialization_type,
    )

    log.info(f"initializing the grid from '{file_path}'")
    grid = driver_init.read_icon_grid(
        path=file_path,
        grid_file=grid_file,
        backend=backend,
        rank=props.rank,
        ser_type=serialization_type,
    )
    log.info(f"reading input fields from '{file_path}'")
    (
        edge_geometry,
        cell_geometry,
        vertical_geometry,
        c_owner_mask,
    ) = driver_init.read_geometry_fields(
        path=file_path,
        grid_file=grid_file,
        vertical_grid_config=config.vertical_grid_config,
        backend=backend,
        rank=props.rank,
        ser_type=serialization_type,
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        _,
    ) = driver_init.read_static_fields(
        path=file_path,
        grid_file=grid_file,
        backend=backend,
        rank=props.rank,
        ser_type=serialization_type,
    )

    log.info("initializing diffusion")
    diffusion_params = diffusion.DiffusionParams(config.diffusion_config)
    exchange = decomposition.create_exchange(props, decomp_info)
    diffusion_granule = diffusion.Diffusion(
        grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        diffusion_metric_state,
        diffusion_interpolation_state,
        edge_geometry,
        cell_geometry,
        exchange=exchange,
        backend=backend,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config.solve_nonhydro_config)

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        grid=grid,
        backend=backend,
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
        second_order_divdamp_factor,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = driver_init.read_initial_state(
        grid=grid,
        cell_param=cell_geometry,
        edge_param=edge_geometry,
        path=file_path,
        backend=backend,
        rank=props.rank,
        experiment_type=experiment_type,
    )
    prognostics_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    time_loop = Driver(
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
        DriverParams(second_order_divdamp_factor=second_order_divdamp_factor),
    )


def icon4py_driver(
    grid_file_path: Annotated[str, typer.Argument(help="Grid file path.")],
    icon4py_driver_backend: Annotated[
        str,
        typer.Argument(
            "--backend",
            help=f"GT4Py backend for running the entire driver. Possible options are: {' / '.join([k for k in model_backends.BACKENDS.keys()])}",
        ),
    ],
    run_path: Annotated[
        str, typer.Option(help="Folder path that holds the output and log files.", default="./")
    ],
    log_level: Annotated[
        str,
        typer.Option(
            help=f"Logging level of log files. Possible options are {' / '.join([k for k in driver_init.LOGGING_LEVELS.keys()])}",
            default=driver_init.LOGGING_LEVELS.keys()[0],
        ),
    ],
    enable_profiling: Annotated[bool, typer.Option(help="Enable profiling.", default=False)],
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
    # checking is run path exists, if not, make a new directory.
    filter_run_path = (
        pathlib.Path(run_path).absolute() if run_path else pathlib.Path("./").absolute()
    )
    os.mkdir(filter_run_path, exist_ok=True)

    time_loop: Driver
    ds: DriverStates
    driver, ds = initialize(
        filter_run_path,
        pathlib.Path(grid_file_path).absolute(),
        log_level,
        icon4py_driver_backend,
    )
    log.info(f"Starting ICON dycore run: {time_loop.simulation_date.isoformat()}")
    log.info(f"input args: grid_path={grid_file_path}")

    log.info("dycore configuring: DONE")
    log.info("time loop: START")

    time_loop.time_integration(
        ds.diffusion_diagnostic,
        ds.solve_nonhydro_diagnostic,
        ds.prognostics,
        ds.prep_advection_prognostic,
        do_prep_adv=False,
        profiling=driver_config.ProfilingConfig() if enable_profiling else None,
    )

    log.info("time loop:  DONE")


if __name__ == "__main__":
    typer.run(icon4py_driver)
