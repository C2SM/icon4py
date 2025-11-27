# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import functools
import logging
import pathlib
import statistics
from collections.abc import Callable

import gt4py.next as gtx
from devtools import Timer
from gt4py.next import config as gtx_config, metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_backends, model_options, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import grid_manager as gm, gridfile, vertical as v_grid
from icon4py.model.common.initialization import jablonowski_williamson_topography
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.standalone_driver import driver_states, driver_utils


log = logging.getLogger(__name__)


# TODO (Chia Rui): I think this should be merged into driver config
# TODO (Chia Rui): config should be moved to a different module when configuration is ready
@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    experiment_name: str
    output_path: pathlib.Path
    profiling_stats: ProfilingStats | None
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = 0.85
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False


@dataclasses.dataclass
class _DerivedFormatter(logging.Formatter):
    style: str
    default_fmt: str
    debug_fmt: str

    _debug_formatter: logging.Formatter = dataclasses.field(init=False)

    def __post_init__(self):
        super().__init__(fmt=self.default_fmt, style=self.style)
        self._debug_formatter = logging.Formatter(
            fmt=self.debug_fmt,
            style=self.style,
        )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.DEBUG:
            return self._debug_formatter.format(record)
        return super().format(record)


class Icon4pyDriver:
    def __init__(
        self,
        config: DriverConfig,
        backend: model_backends.BackendLike,
        grid_manager: gm.GridManager,
        static_field_factories: driver_states.StaticFieldFactories,
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.config = config
        self.backend = backend
        self.grid_manager = grid_manager
        self.static_field_factories = static_field_factories
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule

        self._make_timers()
        self._initialize_timeloop_parameters()
        self._validate_config()
        self._make_logger()
        self._display_setup_in_log_file()

    @functools.cached_property
    def _allocator(self):
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self):
        return data_alloc.import_array_ns(self._allocator)

    @functools.cached_property
    def _concrete_backend(self):
        return model_options.customize_backend(program=None, backend=self.backend)

    def _format_physics_constants(self) -> str:
        consts = PhysicsConstants()
        lines = ["==== Physical Constants ===="]
        for name, value in consts.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            lines.append(f"{name:30s}: {value}")
        return "\n".join(lines)

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
        boundary_line = ["*" * 91]
        icon4py_signature = []
        icon4py_signature += boundary_line
        empty_line = ["*" + 89 * " " + "*"]
        for _ in range(3):
            icon4py_signature += empty_line

        icon4py_signature += [
            "*                                                                ___                      *"
        ]
        icon4py_signature += [
            r"*            -------                                    //      ||   \                    *"
        ]
        icon4py_signature += [
            "*              | |                                     //       ||    |                   *"
        ]
        icon4py_signature += [
            "*              | |       __      _ _        _ _       //  ||    ||___/                    *"
        ]
        icon4py_signature += [
            r"*              | |     //       /   \     |/   \     //_ _||_   ||        \\      //      *"
        ]
        icon4py_signature += [
            r"*              | |    ||       |     |    |     |    --------   ||         \\    //       *"
        ]
        icon4py_signature += [
            r"*              | |     \\__     \_ _/     |     |         ||    ||          \\  //        *"
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
        self._logger.info(f"{icon4py_signature}")

        self._logger.info("===== ICON4Py Driver Configuration =====")
        self._logger.info(f"Experiment name        : {self.config.experiment_name}")
        self._logger.info(f"Time step (dtime)      : {self.config.dtime.total_seconds()} s")
        self._logger.info(f"Number of timesteps    : {self.n_time_steps}")
        self._logger.info(f"Initial ndyn_substeps  : {self.config.ndyn_substeps}")
        self._logger.info(f"Vertical CFL threshold : {self.config.vertical_cfl_threshold}")
        self._logger.info(
            f"Second-order divdamp   : {self.config.apply_extra_second_order_divdamp}"
        )
        self._logger.info(f"Statistics enabled     : {self.config.enable_statistics_output}")
        self._logger.info("")

        self._logger.info("==== Vertical Grid Parameters ====")
        self._logger.info(self.solve_nonhydro._vertical_params)
        self._logger.info(self._format_physics_constants())

    def _make_timers(self) -> None:
        self._timers: dict[str, Timer] = {}
        self._timers["timer_solve_nh_first_step"] = Timer("Solve nh: first time step", dp=6)
        self._timers["timer_solve_nh"] = Timer("Solve nh: after first time step", dp=6)
        self._timers["timer_diffusion_first_step"] = Timer("Diffusion: first time step", dp=6)
        self._timers["timer_diffusion"] = Timer("Diffusion: after first time step", dp=6)

    def _make_logger(self) -> None:
        """
        Create a log file and two log formats for debug and other logging levels. When debug level is used for
        a message, an ascii time stamp and the function name are appended to the beginning of the message.
        """
        self._logger = logging.getLogger("Icon4PyDriver")
        file_handler = logging.FileHandler(
            filename=str(self.config.output_path.joinpath(f"log_{self.config.experiment_name}")),
        )
        logging_formatter = _DerivedFormatter(
            style="{",
            default_fmt="{message}",
            debug_fmt="{asctime} {funcName:<20} : {message}",
        )
        file_handler.setFormatter(logging_formatter)
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)

    def _initialize_timeloop_parameters(self) -> None:
        """
        Initialize parameters needed for running the time loop from start to end date.
        """
        self._n_time_steps: int = int(
            (self.config.end_date - self.config.start_date) / self.config.dtime
        )
        self._dtime_in_seconds: ta.wpfloat = self.config.dtime.total_seconds()
        self._ndyn_substeps_var: int = self.config.ndyn_substeps
        self._max_ndyn_substeps: int = self.config.ndyn_substeps + 7
        self._substep_timestep: ta.wpfloat = ta.wpfloat(
            self._dtime_in_seconds / self._ndyn_substeps_var
        )
        self._elapse_time_in_seconds: ta.wpfloat = ta.wpfloat("0.0")

        # current simulation date
        self._simulation_date: datetime.datetime = self.config.start_date

        self._is_first_step_in_simulation: bool = True

        self._cfl_watch_mode: bool = False

    def _validate_config(self) -> None:
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    def _update_ndyn_substeps(self, new_ndyn_substeps: int) -> None:
        self._ndyn_substeps_var = new_ndyn_substeps
        self._substep_timestep = ta.wpfloat(self._dtime_in_seconds / self._ndyn_substeps_var)

    def re_initialization(self) -> None:
        """
        Re-initialize the time step, number of substeps, elapsed time, simulation date, and other
        time integration parameters for re-run.
        """
        self._logger.info("Reinitialize the driver")
        self._initialize_timeloop_parameters()

    @property
    def simulation_date(self) -> datetime.datetime:
        return self._simulation_date

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
        self._elapse_time_in_seconds += self.config.dtime.total_seconds()

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
            rho_arg_max, max_rho = driver_utils.find_maximum_from_field(
                self._xp, prognostic_states.rho
            )
            vn_arg_max, max_vn = driver_utils.find_maximum_from_field(
                self._xp, prognostic_states.vn
            )
            w_arg_max, max_w = driver_utils.find_maximum_from_field(self._xp, prognostic_states.w)

            def _determine_sign(input_number: float) -> str:
                return " " if input_number >= 0.0 else "-"

            rho_sign = _determine_sign(max_rho)
            vn_sign = _determine_sign(max_vn)
            w_sign = _determine_sign(max_w)

            self._logger.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self._ndyn_substeps_var:3d} == "
                f"MAX RHO: {rho_sign}{abs(max_rho):.5e} at lvl {rho_arg_max[1]:4d}, MAX VN: {vn_sign}{abs(max_vn):.5e} at lvl {vn_arg_max[1]:4d}, MAX W: {w_sign}{abs(max_w):.5e} at lvl {w_arg_max[1]:4d}"
            )
        else:
            self._logger.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self._ndyn_substeps_var:3d}"
            )

    def _compute_total_mass_and_energy(
        self, prognostic_states: prognostics.PrognosticState
    ) -> None:
        if self.config.enable_statistics_output:
            rho_ndarray = prognostic_states.rho.ndarray
            cell_volume_ndarray = self.grid_manager.geometry_fields[
                gridfile.GeometryName.CELL_AREA.value
            ].ndarray
            cell_thickness_ndarray = self.static_field_factories.metrics_field_source.get(
                metrics_attr.DDQZ_Z_FULL
            ).ndarray
            total_mass = self._xp.sum(rho_ndarray * cell_volume_ndarray * cell_thickness_ndarray)
            self._logger.info(f"TOTAL MASS: {total_mass:.10e}, TOTAL ENERGY")

    def _compute_mean_at_final_time_step(
        self, prognostic_states: prognostics.PrognosticState
    ) -> None:
        if self.config.enable_statistics_output:
            rho_ndarray = prognostic_states.rho.ndarray
            vn_ndarray = prognostic_states.vn.ndarray
            w_ndarray = prognostic_states.w.ndarray
            theta_v_ndarray = prognostic_states.theta_v.ndarray
            exner_ndarray = prognostic_states.exner.ndarray
            interface_physical_height_ndarray = (
                self.solve_nonhydro._vertical_params.interface_physical_height.ndarray
            )
            self._logger.info("")
            self._logger.info(
                "Global mean of    rho         vn           w          theta_v     exner      at model levels:"
            )
            for k in range(rho_ndarray.shape[1]):
                self._logger.info(
                    f"{interface_physical_height_ndarray[k]:12.3f}: {self._xp.mean(rho_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(vn_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(w_ndarray[:, k+1]):.5e} "
                    f"{self._xp.mean(theta_v_ndarray[:, k]):.5e} "
                    f"{self._xp.mean(exner_ndarray[:, k]):.5e} "
                )

    def _show_timer_report(
        self,
    ) -> None:
        for timer_name, timer in self._timers.items():
            try:
                timer_summary = timer.summary(False)
                self._logger.info(
                    f"{timer_name} timer summary: "
                    f"times={len(timer_summary)}, "
                    f"mean={statistics.mean(timer_summary):0.8f}s, "
                    f"stdev={statistics.stdev(timer_summary) if len(timer_summary) > 1 else 0:0.8f}s, "
                    f"min={min(timer_summary):0.8f}s, "
                    f"max={max(timer_summary):0.8f}s"
                )
            except RuntimeError:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
                self._logger.info(f"Timer {timer_name} has not started")

    def time_integration(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ) -> None:
        self._logger.debug(
            f"starting time loop for dtime = {self._dtime_in_seconds} s, substep_timestep = {self._substep_timestep} s, n_timesteps = {self.n_time_steps}, substep_timestep = {self.substep_timestep}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        actual_starting_time = datetime.datetime.now()

        for time_step in range(self._n_time_steps):
            if self.config.profiling_stats is not None:
                if not self.config.profiling_stats.skip_first_timestep or time_step > 0:
                    gtx_config.COLLECT_METRICS_LEVEL = (
                        self.config.profiling_stats.gt4py_metrics_level
                    )

            self._logger.info(
                f"\n"
                f"simulation date : {self._simulation_date}, at run timestep : {time_step}, with actual time spent: {(datetime.datetime.now() - actual_starting_time).total_seconds()}"
                f"\n"
            )

            self._next_simulation_date()

            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv,
                do_prep_adv,
            )
            device_utils.sync(self._concrete_backend)

            self._is_first_step_in_simulation = False

            self._adjust_ndyn_substeps_var(solve_nonhydro_diagnostic_state)

            # TODO(OngChia): simple IO enough for JW test

        self._compute_mean_at_final_time_step(prognostic_states.current)

        self._show_timer_report()
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
        self._logger.debug(f"Running {self.solve_nonhydro.__class__}")
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_states,
            prep_adv,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self._logger.debug(f"Running {self.diffusion.__class__}")
            timer_diffusion = (
                self._timers["timer_diffusion_first_step"]
                if self._is_first_step_in_simulation
                else self._timers["timer_diffusion"]
            )
            timer_diffusion.start()
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self._dtime_in_seconds,
            )
            timer_diffusion.capture()

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

        timer_solve_nh = (
            self._timers["timer_solve_nh_first_step"]
            if self._is_first_step_in_simulation
            else self._timers["timer_solve_nh"]
        )
        for dyn_substep in range(self._ndyn_substeps_var):
            self._compute_statistics(dyn_substep, prognostic_states.current)

            self._update_time_levels_for_velocity_tendencies(
                solve_nonhydro_diagnostic_state,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_initial_timestep=self._is_first_step_in_simulation,
            )

            timer_solve_nh.start()
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv=prep_adv,
                second_order_divdamp_factor=self._update_spinup_second_order_divergence_damping(),
                dtime=self._substep_timestep,
                ndyn_substeps_var=self._ndyn_substeps_var,
                at_initial_timestep=self._is_first_step_in_simulation,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )
            timer_solve_nh.capture()

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()
        self._compute_total_mass_and_energy(dyn_substep, prognostic_states.next)

        # TODO(OngChia): compute airmass for prognostic_state here

    # watch_mode is true if step is <= 1 or cfl already near or exceeding threshold.
    # omit spinup feature and the option that if the model starts from IFS or COSMO data
    # horizontal cfl is not ported
    def _adjust_ndyn_substeps_var(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
    ) -> None:
        # TODO (Chia Rui): perform a global max operation in multinode run
        global_max_vertical_cfl = solve_nonhydro_diagnostic_state.max_vertical_cfl[()]

        if (
            global_max_vertical_cfl > ta.wpfloat("0.81") * self.config.vertical_cfl_threshold
            and not self._cfl_watch_mode
        ):
            self._logger.warning(
                "High CFL number for vertical advection in dynamical core, entering watch mode"
            )
            self._cfl_watch_mode = True

        if self._cfl_watch_mode:
            substep_fraction = ta.wpfloat(self._ndyn_substeps_var / self.config.ndyn_substeps)
            if (
                global_max_vertical_cfl * substep_fraction
                > ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            ):
                self._logger.warning(
                    f"Maximum vertical CFL number {global_max_vertical_cfl} is close to critical threshold"
                )

            vertical_cfl_threshold_for_increment = self.config.vertical_cfl_threshold
            vertical_cfl_threshold_for_decrement = (
                ta.wpfloat("0.9") * self.config.vertical_cfl_threshold
            )

            if global_max_vertical_cfl > vertical_cfl_threshold_for_increment:
                if self._xp.isfinite(global_max_vertical_cfl):
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
                else:
                    self._logger.warning(
                        f"WARNING: max cfl {global_max_vertical_cfl} is not a number! Number of substeps is set to the max value! "
                    )
                    new_ndyn_substeps_var = self._max_ndyn_substeps
                self._update_ndyn_substeps(new_ndyn_substeps_var)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                self._logger.warning(
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
                self._logger.warning(
                    f"The number of dynamics substeps is decreased to {self._ndyn_substeps_var}"
                )

                if (
                    self._ndyn_substeps_var == self.config.ndyn_substeps
                    and global_max_vertical_cfl
                    < ta.wpfloat("0.76") * self.config.vertical_cfl_threshold
                ):
                    self._logger.warning(
                        "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
                    )
                    self._cfl_watch_mode = False

        # reset max_vertical_cfl to zero
        solve_nonhydro_diagnostic_state.max_vertical_cfl = data_alloc.scalar_like_array(
            0.0, self._allocator
        )

    def _update_spinup_second_order_divergence_damping(self) -> ta.wpfloat:
        if self.config.apply_extra_second_order_divdamp:
            fourth_order_divdamp_factor = self.solve_nonhydro._config.fourth_order_divdamp_factor
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


# TODO (Chia Rui): this should be replaced by real configuration reader when the configuration PR is merged
def _read_config(
    output_path: pathlib.Path,
    enable_profiling: bool,
) -> tuple[
    DriverConfig,
    v_grid.VerticalGridConfig,
    diffusion.DiffusionConfig,
    solve_nh.NonHydrostaticConfig,
]:
    vertical_grid_config = v_grid.VerticalGridConfig(
        num_levels=35,
        rayleigh_damping_height=45000.0,
    )

    diffusion_config = diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        hdiff_temp=False,
        n_substeps=5,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=10.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=False,
        velocity_boundary_diffusion_denom=200.0,
    )

    nonhydro_config = solve_nh.NonHydrostaticConfig(
        fourth_order_divdamp_factor=0.0025,
    )

    profiling_stats = ProfilingStats() if enable_profiling else None

    driver_config = DriverConfig(
        experiment_name="Jablonowski_Williamson",
        output_path=output_path,
        dtime=datetime.timedelta(seconds=300.0),
        end_date=datetime.datetime(1, 1, 1, 1, 0, 0),
        apply_extra_second_order_divdamp=False,
        ndyn_substeps=5,
        vertical_cfl_threshold=ta.wpfloat("0.85"),
        enable_statistics_output=True,
        profiling_stats=profiling_stats,
    )

    return (
        driver_config,
        vertical_grid_config,
        diffusion_config,
        nonhydro_config,
    )


def initialize_driver(
    configuration_file_path: pathlib.Path,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    log_level: str,
    backend: model_backends.BackendLike,
) -> Icon4pyDriver:
    """
    Initialize the driver:
    - load the configuration
    - load the grid manager and decomposition info
    - load the topography (eventually all external parameters)
    - create the static field factories
    - initialize the components selected by the configuration (diffusion and solve_nh)
    - create the driver object
    Args:
        configuration_file_path: path to the configuration file
        output_path: path where to store the simulation output
        grid_file_path: path of the grid file
        log_level: logging level
        backend: GT4Py backend-like
    Returns:
        Driver: driver object
    """

    allocator = model_backends.get_allocator(backend)

    driver_config, vertical_grid_config, diffusion_config, solve_nh_config = _read_config(
        output_path=output_path,
        enable_profiling=False,
    )
    parallel_props = decomposition_defs.get_processor_properties(
        decomposition_defs.get_runtype(with_mpi=False)
    )
    driver_utils.configure_logging(
        logging_level=log_level,
        processor_procs=parallel_props,
    )

    log.info(f"initializing the grid manager from '{grid_file_path}'")
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
    )

    log.info("creating the decomposition info")

    decomposition_info = driver_utils.create_decomposition_info(
        grid_manager=grid_manager,
        allocator=allocator,
    )
    exchange = decomposition_defs.create_exchange(parallel_props, decomposition_info)

    log.info("initializing the vertical grid")
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
    )

    log.info("initializing the JW topography")
    cell_topography = jablonowski_williamson_topography.jablonowski_williamson_topography(
        cell_lat=grid_manager.coordinates[dims.CellDim]["lat"].ndarray,
        u0=35.0,
        array_ns=data_alloc.import_array_ns(allocator=allocator),
    )

    log.info("initializing the static-field factories")
    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid_manager,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field((dims.CellDim,), data=cell_topography, allocator=allocator),
        backend=backend,
    )

    log.info("initializing granules")
    (
        diffusion_granule,
        solve_nonhydro_granule,
    ) = driver_utils.initialize_granules(
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        diffusion_config=diffusion_config,
        solve_nh_config=solve_nh_config,
        static_field_factories=static_field_factories,
        exchange=exchange,
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),
            allocator=allocator,  # type: ignore[arg-type]
        ),
        backend=backend,
    )

    icon4py_driver = Icon4pyDriver(
        config=driver_config,
        backend=backend,
        grid_manager=grid_manager,
        static_field_factories=static_field_factories,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )

    return icon4py_driver
