# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import statistics
from typing import TYPE_CHECKING, Any, NamedTuple

import devtools

import icon4py.model.common.utils as common_utils
from icon4py.model.common import dimension as dims, time, type_alias as ta
from icon4py.model.common.components import framework as fw, states
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation
from icon4py.model.common.states import static_fields
from icon4py.model.driver import config as driver_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing
    from gt4py.next import backend as gtx_backend


log = logging.getLogger(__name__)


class DriverStates(NamedTuple):
    """
    Initialized states for the driver run.

    Attributes:
        prep_advection_prognostic: Fields collecting data for advection during the solve nonhydro timestep.
        dycore_forcing: The tendencies and increments the dycore reads.
        dycore_diagnostics: The diagnostics the dycore carries between the substeps.
        diffusion_diagnostic: Initial state for diffusion diagnostic variables.
        tracer_advection_diagnostic: Initial state for tracer advection diagnostic variables.
        prognostics: Initial state for prognostic variables (double buffered, swapped
            once per dynamics substep).
        tracers: Initial tracer state (double buffered, swapped once per time step).
        diagnostic: Initial state for global diagnostic variables.
    """

    prep_advection_prognostic: states.PrepAdvection | None
    dycore_forcing: states.DycoreForcing | None
    dycore_diagnostics: states.DycoreDiagnostics | None
    diffusion_diagnostic: states.DiffusionDiagnostics | None
    tracer_advection_diagnostic: states.AdvectionDiagnostics | None
    prognostics: common_utils.TimeStepPair[states.PrognosticState]
    tracers: common_utils.TimeStepPair[states.TracerState]
    diagnostic: states.Diagnostics


class ModelTimeVariables:
    """
    Runtime time/date variables derived from config at initialisation.
    """

    simulation_current_datetime: time.AbsoluteTime
    simulation_start_datetime: time.AbsoluteTime
    simulation_end_datetime: time.AbsoluteTime
    n_time_steps: time.NumTimeSteps
    dtime: time.RelativeTime
    ndyn_substeps_var: int
    max_ndyn_substeps: int
    elapsed_time_in_seconds: ta.wpfloat
    is_first_step_in_simulation: bool
    cfl_watch_mode: bool

    def __init__(self, config: driver_config.DriverConfig) -> None:
        self._init_from_config(config)

    def _init_from_config(self, config: driver_config.DriverConfig) -> None:
        self.simulation_start_datetime = config.start_of_simulation
        # The time loop starts at the beginning of the simulation, unless restarting.
        self.simulation_current_datetime = config.start_of_timestepping
        match config.end_of_simulation:
            case time.NumTimeSteps() as n:
                self.n_time_steps = n
                requested_simulation_end_datetime = config.start_of_timestepping + n * config.dtime
            case time.RelativeTime() as relative:
                self.n_time_steps = int(relative / config.dtime)
                requested_simulation_end_datetime = config.start_of_timestepping + relative
            case time.AbsoluteTime() as absolute:
                self.n_time_steps = int((absolute - config.start_of_timestepping) / config.dtime)
                requested_simulation_end_datetime = absolute

        self.simulation_end_datetime = (
            config.start_of_timestepping + self.n_time_steps * config.dtime
        )

        if requested_simulation_end_datetime != self.simulation_end_datetime:
            raise ValueError(
                f"The requested end_of_simulation is not an integer number of time steps. Requested: {requested_simulation_end_datetime}, computed: {self.simulation_end_datetime}"
            )

        self.dtime = config.dtime
        # measured from the beginning of the simulation, also when restarting (just for consistency with fortran)
        self.elapsed_time_in_seconds = ta.wpfloat(
            (config.start_of_timestepping - config.start_of_simulation).total_seconds()
        )
        self.ndyn_substeps_var = config.ndyn_substeps
        self.max_ndyn_substeps = config.ndyn_substeps + 7
        self.is_first_step_in_simulation = (
            config.start_of_timestepping == config.start_of_simulation
        )
        self.cfl_watch_mode = False

        if self.n_time_steps <= 0:
            raise ValueError("n_time_steps must be positive.")

    @functools.cached_property
    def dtime_in_seconds(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime.total_seconds())

    @property
    def substep_timestep(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime_in_seconds / self.ndyn_substeps_var)

    @property
    def elapsed_time_at_step_midpoint_in_seconds(self) -> ta.wpfloat:
        """
        Elapsed time at the middle of the current time step.

        elapsed_time_global = (jstep - 0.5) * dtime in mo_nh_stepping.f90, with a
        one-based jstep. 'advance_simulation_datetime' is called before the step is
        integrated, so 'elapsed_time_in_seconds' is already at the end of it.
        """
        return ta.wpfloat(self.elapsed_time_in_seconds - 0.5 * self.dtime_in_seconds)

    def advance_simulation_datetime(self) -> None:
        self.simulation_current_datetime += self.dtime
        self.elapsed_time_in_seconds += self.dtime_in_seconds

    def update_ndyn_substeps(self, new_ndyn_substeps: int) -> None:
        self.ndyn_substeps_var = new_ndyn_substeps

    def update_cfl_watch_mode(self, mode: bool) -> None:
        self.cfl_watch_mode = mode

    def reset(self, config: driver_config.DriverConfig) -> None:
        """
        Re-initialize all time-integration-related runtime values from the given config.
        """
        self._init_from_config(config)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"simulation_current_datetime={self.simulation_current_datetime!r}, "
            f"n_time_steps={self.n_time_steps}, "
            f"dtime={self.dtime}, "
            f"is_first_step={self.is_first_step_in_simulation})"
        )


class DriverTimers(enum.Enum):
    SOLVE_NH_FIRST_STEP = "solve_nh_first_step"
    SOLVE_NH = "solve_nh"
    DIFFUSION_FIRST_STEP = "diffusion_first_step"
    DIFFUSION = "diffusion"
    #: the diagnostics for output (capture steps only)
    OUTPUT_ASSEMBLE = "output_assemble"
    #: the IO component: host transfer, gather/halo stripping and file writing at capture
    #: steps; a near-zero sample otherwise
    OUTPUT_STORE = "output_store"


@dataclasses.dataclass
class TimerCollection:
    timer_names: dataclasses.InitVar[list[str]]
    timers: dict[str, devtools.Timer] = dataclasses.field(init=False)

    def __post_init__(self, timer_names: str | list[str]) -> None:
        self.timers = {}
        self.add_timers(timer_names)

    def add_timers(self, timer_names: str | list[str]) -> None:
        for timer in timer_names:
            assert timer not in self.timers, f"Timer '{timer}' is already defined."
            self.timers[timer] = devtools.Timer(timer, dp=6, verbose=False)

    def show_timer_report(
        self,
        total_wall_time: float,
    ) -> None:
        """Log the per-timer statistics.

        ``total_wall_time`` is the wall-clock duration of the whole simulation, in
        seconds; an extra column reports each timer's cumulative time as a percentage
        of it. The percentages need not sum to 100: the difference is time spent
        outside any timer.
        """
        log.info("===== ICON4Py timer report =====")
        wall_time = total_wall_time if total_wall_time > 0 else None
        table_titles = (
            f"|{'timer name':^30}|"
            f"{'no. of times called':^23}|"
            f"{'mean time (s)':^23}|"
            f"{'std. deviation (s)':^23}|"
            f"{'min time (s)':^23}|"
            f"{'max time (s)':^23}|"
            f"{'% of wall time':^23}|"
        )
        log.info(table_titles)
        log.info("-" * len(table_titles))
        timed_total = 0.0
        for timer_name, timer in self.timers.items():
            times = []
            for r in timer.results:
                if not r.finish:
                    r.capture()
                times.append(r.elapsed())
            if len(times) > 0:
                timed_total += sum(times)
                share = f"{100 * sum(times) / wall_time:.2f}" if wall_time is not None else "n/a"
                log.info(
                    f"|{timer_name:^30}|"
                    f"{len(times):^23}|"
                    f"{statistics.mean(times):^23.8f}|"
                    f"{statistics.stdev(times) if len(times) > 1 else 0:^23.8f}|"
                    f"{min(times):^23.8f}|"
                    f"{max(times):^23.8f}|"
                    f"{share:^23}|"
                )
            else:
                log.info(
                    f"|{timer_name:^30}|{'not started':^23}|{'':^23}|{'':^23}|{'':^23}|{'':^23}|{'':^23}|"
                )
        if wall_time is not None:
            log.info("-" * len(table_titles))
            log.info(f"total wall-clock time of the simulation: {wall_time:.8f} s")
            timed_share = 100 * timed_total / wall_time
            log.info(
                f"timed regions total: {timed_total:.8f} s  "
                f"({timed_share:.2f}% of wall time; {100 - timed_share:.2f}% untimed)"
            )


def assemble_driver_states(
    *,
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
    backend: gtx_backend.Backend[Any] | None,
    exchange: decomposition_defs.ExchangeRuntime,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: states.PrognosticState,
    tracer_state_now: states.TracerState,
    diagnostic_state: states.Diagnostics,
    experiment_config: driver_config.ExperimentConfig,
    dycore_forcing: states.DycoreForcing | None,
    dycore_diagnostics: states.DycoreDiagnostics | None,
    prep_adv: states.PrepAdvection | None,
) -> DriverStates:
    prognostic_states = common_utils.TimeStepPair(
        prognostic_state_now, fw.copy(prognostic_state_now, allocator)
    )
    tracer_states = common_utils.TimeStepPair(
        tracer_state_now, fw.copy(tracer_state_now, allocator)
    )

    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    rbf_vec_coeff_c1 = static_fields.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C1)
    rbf_vec_coeff_c2 = static_fields.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C2)

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=prognostic_states.current.vn.data,
        ptr_coeff_1=rbf_vec_coeff_c1,
        ptr_coeff_2=rbf_vec_coeff_c2,
        p_u_out=diagnostic_state.u.data,
        p_v_out=diagnostic_state.v.data,
        horizontal_start=end_cell_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider=grid.connectivities,
    )
    exchange.exchange(dims.CellDim, diagnostic_state.u.data, diagnostic_state.v.data)

    diffusion_enabled = experiment_config.diffusion is not None
    tracer_advection_enabled = experiment_config.tracer_advection is not None

    diffusion_diagnostic_state = (
        fw.allocate(states.DiffusionDiagnostics, grid, allocator) if diffusion_enabled else None
    )
    tracer_advection_diagnostic_state = (
        fw.allocate(states.AdvectionDiagnostics, grid, allocator)
        if tracer_advection_enabled
        else None
    )

    return DriverStates(
        prep_advection_prognostic=prep_adv,
        dycore_forcing=dycore_forcing,
        dycore_diagnostics=dycore_diagnostics,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        tracers=tracer_states,
        diagnostic=diagnostic_state,
    )
