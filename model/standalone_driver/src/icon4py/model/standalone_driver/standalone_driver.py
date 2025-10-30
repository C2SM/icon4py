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
from collections.abc import Callable
from typing import Annotated, NamedTuple

import typer
from devtools import Timer
from gt4py.next import config as gtx_config, metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import model_backends, type_alias as ta
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

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.DEBUG:
            return self._debug_formatter.format(record)
        else:
            return super().format(record)


class Icon4pyDriver:
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
        self._log = logging.getLogger(self._full_name)

        self._initialize_timeloop_parameters()
        self._validate_config()
        self._filter_log()
        self._display_setup_in_log_file()

    def _display_setup_in_log_file(self) -> None:
        """
        Print out icon4py signature and some important information of the initial setup to the log file.

                                                                ___                      
            -------                                    //      ||   \                    
              | |                                     //       ||    |                   
              | |       __      _ _        _ _       //  ||    ||___/                    
              | |     //       /   \     |/   \     //_ _||_   ||        \\      //      
              | |    ||       |     |    |     |    --------   ||         \\    //       
              | |     \\__     \_ _/     |     |         ||    ||          \\  //        
            -------                                                           //         
                                                                             //          
                                                                = = = = = = //           
        """
        boundary_line = "*+*" * 33
        icon4py_signature = []
        icon4py_signature += boundary_line
        empty_line = "*" + 97 * " " + "+"
        for _ in range(3):
            icon4py_signature += empty_line

        icon4py_signature += [
            "*                                                                ___                      *"
        ]
        icon4py_signature += [
            "*            -------                                    //      ||   \                    *"
        ]
        icon4py_signature += [
            "*              | |                                     //       ||    |                   *"
        ]
        icon4py_signature += [
            "*              | |       __      _ _        _ _       //  ||    ||___/                    *"
        ]
        icon4py_signature += [
            "*              | |     //       /   \     |/   \     //_ _||_   ||        \\      //      *"
        ]
        icon4py_signature += [
            "*              | |    ||       |     |    |     |    --------   ||         \\    //       *"
        ]
        icon4py_signature += [
            "*              | |     \\__     \_ _/     |     |         ||    ||          \\  //        *"
        ]
        icon4py_signature += [
            "*            -------                                                           //         *"
        ]
        icon4py_signature += [
            "*                                                                             //          *"
        ]
        icon4py_signature += [
            "*                                                                = = = = = = //           *"
        ]

        for _ in range(3):
            icon4py_signature += empty_line
        icon4py_signature += boundary_line
        icon4py_signature = "\n".join(icon4py_signature)
        self._log.info(f"{icon4py_signature}")
        self._log.info(self.solve_nonhydro._vertical_params)

    def _filter_log(self) -> None:
        """
        Create a log file and two log formats for debug and other logging levels. When debug level is used for
        a message, an ascii time stamp and the function name are appended to the beginning of the message.
        """
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

    def _initialize_timeloop_parameters(self) -> None:
        """
        Initialize parameters needed for running the time loop from start to end date.
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

    def _validate_config(self) -> None:
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    def _update_ndyn_substeps(self, new_ndyn_substeps: int) -> None:
        self._ndyn_substeps_var = new_ndyn_substeps

    def re_initialization(self) -> None:
        """
        Re-initialize the time step, number of substeps, elapsed time, simulation date, and other
        time integration parameters for re-run.
        """
        self._initialize_timeloop_parameters()
        self._validate_config()

    @property
    def first_step_in_simulation(self) -> bool:
        return self._is_first_step_in_simulation

    def _is_last_substep(self, step_nr: int) -> bool:
        return step_nr == (self._ndyn_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int) -> bool:
        return step_nr == 0

    def _next_simulation_date(self) -> None:
        self._simulation_date += self.config.dtime
        self._elapse_time_in_seconds += self.config.dtime

    @property
    def n_time_steps(self) -> int:
        return self._n_time_steps

    @property
    def substep_timestep(self) -> int:
        return self._substep_timestep

    def _full_name(self, func: Callable) -> str:
        return f"{self.__class__.__name__}:{func.__name__}"

    def _compute_statistics(
        self, current_dyn_substep: int, prognostic_states: prognostics.PrognosticState
    ) -> None:
        """
        Compute relevant statistics of prognostic variables at the beginning of every time step. The statistics include:
        absolute maximum value of rho, vn, and w, as well as the levels at which their maximum value is found.
        """
        if self.config.enable_statistics_output:
            # TODO (Chia Rui): Do global max when multinode is ready
            rho_arg_max = self._xp.abs(prognostic_states.current.rho.ndarray).max()
            vn_arg_max = self._xp.abs(prognostic_states.current.vn.ndarray).max()
            w_arg_max = self._xp.abs(prognostic_states.current.w.ndarray).max()
            self._log.info(
                f"substep / n_substeps : {current_dyn_substep} / {self._ndyn_substeps_var} == "
                f" MAX RHO, VN, and W: {prognostic_states.current.rho.ndarray[rho_arg_max]:.5e} / {prognostic_states.current.vn.ndarray[vn_arg_max]:.5e} / {prognostic_states.current.w.ndarray[w_arg_max]:.5e}, "
                f"at levels: {self._xp.unravel_index(rho_arg_max, prognostic_states.current.rho.ndarray.shape)[1]} / {self._xp.unravel_index(vn_arg_max, prognostic_states.current.vn.ndarray.shape)[1]} / {self._xp.unravel_index(w_arg_max, prognostic_states.current.w.ndarray.shape)[1]}"
            )
        else:
            self._log.info(
                f"substep / n_substeps : {current_dyn_substep} / {self._ndyn_substeps_var}"
            )

    def time_integration(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ) -> None:
        self._log.debug(
            f"starting time loop for dtime = {self._dtime_in_seconds} s, substep_timestep = {self._substep_timestep} s, n_timesteps = {self.n_time_steps}, substep_timestep = {self.substep_timestep}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        timer_first_timestep = Timer("Driver: first time step", dp=6)
        timer_after_first_timestep = Timer("Driver: after first time step", dp=6)

        for time_step in range(self._n_time_steps):
            timer = timer_first_timestep if time_step == 0 else timer_after_first_timestep
            if self.config.profiling_stats is not None:
                if not self.config.profiling_stats.skip_first_timestep or time_step > 0:
                    gtx_config.COLLECT_METRICS_LEVEL = (
                        self.config.profiling_stats.gt4py_metrics_level
                    )

            self._log.debug(
                f"simulation date : {self._simulation_date}, at run timestep : {time_step}"
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

            self._adjust_ndyn_substeps_var(solve_nonhydro_diagnostic_state)

            # TODO(OngChia): simple IO enough for JW test

        timer_first_timestep.summary(True)
        if self.n_time_steps > 1:  # in case only one time step was run
            timer_after_first_timestep.summary(True)
        if (
            self.config.profiling_stats is not None
            and self.config.profiling_stats.gt4py_metrics_level > gtx_metrics.DISABLED
        ):
            print(gtx_metrics.dumps())
            gtx_metrics.dump_json(self.config.profiling_stats.gt4py_metrics_output_file)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ):
        self._log.debug(f"Running {self.solve_nonhydro.__class__}")
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_states,
            prep_adv,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self._log.debug(f"Running {self.diffusion.__class__}")
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self._dtime_in_seconds,
            )

        prognostic_states.swap()

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
            self._compute_statistics(dyn_substep, prognostic_states)

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
    ) -> None:
        skip_checking_watch_mode_quit = False
        # TODO (Chia Rui): perform a global max operation in multinode run
        global_max_vertical_cfl = solve_nonhydro_diagnostic_state.max_vertical_cfl

        if (
            global_max_vertical_cfl > ta.wpfloat("0.81") * self.config.vertical_cfl_threshold
            and not self._cfl_watch_mode
        ):
            self._log.info(
                "High CFL number for vertical advection in dynamical core, entering watch mode"
            )
            self._cfl_watch_mode = True

        if self._cfl_watch_mode:
            substep_fraction = ta.wpfloat(self._ndyn_substeps_var / self.config.ndyn_substeps)
            if (
                global_max_vertical_cfl * substep_fraction
                > ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            ):
                self._log.info(
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
                self._log.info(
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
                self._log.info(
                    f"The number of dynamics substeps is decreaed to {self._ndyn_substeps_var}"
                )
                skip_checking_watch_mode_quit = True
        if (
            not skip_checking_watch_mode_quit
            and self._ndyn_substeps_var == self.config.ndyn_substeps
            and global_max_vertical_cfl < ta.wpfloat("0.76") * self.config.vertical_cfl_threshold
        ):
            self._log.info(
                "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
            )
            self._cfl_watch_mode = False

        # reset max_vertical_cfl to zero
        solve_nonhydro_diagnostic_state.max_vertical_cfl = ta.wpfloat("0.0")

    def _update_spinup_second_order_divergence_damping(
        self, fourth_order_divdamp_factor: ta.wpfloat
    ) -> ta.wpfloat:
        if self.config.apply_extra_second_order_divdamp:
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


def initialize(
    configuration_file_path: pathlib.Path,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    log_level: str,
    backend: str,
) -> tuple[Icon4pyDriver, DriverStates]:
    """
    Initialize the driver run.

    This function does the following:
    - load configuration
    - load grid information
    - initialize components: diffusion and solve_nh
    - load diagnostic and prognostic variables (serialized data)
    - setup the time loop

    Parameters:
        grid_file_path: Path to the serialized data.
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
    driver_init.configure_logging(output_path, log_level, "Jablownoski-Williamson", parallel_props)

    log.info("initialize parallel runtime")
    log.info("reading configuration: experiment Jablownoski-Williamson")
    driver_config, vertical_grid_config, diffusion_config, solve_nh_config = (
        driver_configure.read_config(
            configuration_file_path=configuration_file_path,
            output_path=output_path,
            grid_file_path=grid_file_path,
            backend=backend,
        )
    )

    log.info(f"initializing the grid from '{grid_file_path}'")
    (grid, decomposition_info) = driver_init.create_mesh(
        grid_file=grid_file_path,
        vertical_grid_config=vertical_grid_config,
        backend=backend,
    )
    vertical_grid = driver_init.create_vertical_grid(
        vertical_grid_config=vertical_grid_config,
        backend=backend,
    )

    geometry_field_source = driver_init.create_geometry_factory(
        mesh=grid,
        decomposition_info=decomposition_info,
        backend=backend,
    )

    topo_c = driver_init.create_topography(
        geometry_field_source=geometry_field_source,
        backend=backend,
    )

    (
        interpolation_field_source,
        metrics_field_source,
    ) = driver_init.create_interpolation_metrics_factories(
        mesh=grid,
        decomposition_info=decomposition_info,
        geometry_field_source=geometry_field_source,
        vertical_grid=vertical_grid,
        topo_c=topo_c,
        backend=backend,
    )

    log.info(f"reading input fields from '{grid_file_path}'")

    log.info("initializing diffusion")
    exchange = decomposition.create_exchange(parallel_props, decomposition_info)
    (
        diffusion_granule,
        solve_nonhydro_granule,
    ) = driver_init.initialize_granule(
        mesh=grid,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        diffusion_config=diffusion_config,
        solve_nh_config=solve_nh_config,
        geometry_field_source=geometry_field_source,
        interpolation_field_source=interpolation_field_source,
        metrics_field_source=metrics_field_source,
        exchange=exchange,
        backend=backend,
    )

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = driver_init.read_initial_state(
        grid=grid,
        geometry_field_source=geometry_field_source,
        path=grid_file_path,
        backend=backend,
        rank=parallel_props.rank,
    )

    icon4py_driver = Icon4pyDriver(
        run_config=driver_config,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )
    prognostics_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    return (
        icon4py_driver,
        DriverStates(
            prep_advection_prognostic=prep_adv,
            solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
            diffusion_diagnostic=diffusion_diagnostic_state,
            prognostics=prognostics_states,
            diagnostic=diagnostic_state,
        ),
    )


# TODO (Chia Rui): Ultimately, these arguments and options should be read from a config file and the only argument should be the path to the config file
def run_icon4py_driver(
    configuration_file_path: Annotated[str, typer.Argument(help="Configuration file path.")],
    grid_file_path: Annotated[str, typer.Argument(help="Grid file path.")],
    icon4py_driver_backend: Annotated[
        str,
        typer.Argument(
            "--backend",
            help=f"GT4Py backend for running the entire driver. Possible options are: {' / '.join([k for k in model_backends.BACKENDS])}",
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Folder path that holds the output and log files.", default="./")
    ],
    log_level: Annotated[
        str,
        typer.Option(
            help=f"Logging level of log files. Possible options are {' / '.join([k for k in driver_init._LOGGING_LEVELS])}",
            default=next(iter(driver_init._LOGGING_LEVELS.keys())),
        ),
    ],
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
    filter_output_path = (
        pathlib.Path(output_path).absolute() if output_path else pathlib.Path("./").absolute()
    )
    pathlib.Path.mkdir(filter_output_path, exist_ok=True)

    icon4py_driver: Icon4pyDriver
    ds: DriverStates
    icon4py_driver, ds = initialize(
        pathlib.Path(configuration_file_path).absolute(),
        filter_output_path,
        pathlib.Path(grid_file_path).absolute(),
        log_level,
        icon4py_driver_backend,
    )
    log.info(f"Starting ICON dycore run: {icon4py_driver.simulation_date.isoformat()}")
    log.info(f"input args: grid_path={grid_file_path}")

    log.info("dycore configuring: DONE")
    log.info("time loop: START")

    icon4py_driver.time_integration(
        ds.diffusion_diagnostic,
        ds.solve_nonhydro_diagnostic,
        ds.prognostics,
        ds.prep_advection_prognostic,
        do_prep_adv=False,
    )

    log.info("time loop:  DONE")


if __name__ == "__main__":
    typer.run(run_icon4py_driver)
