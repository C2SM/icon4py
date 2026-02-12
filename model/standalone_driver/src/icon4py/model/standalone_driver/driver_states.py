# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import enum
import functools
import logging
import statistics
from typing import NamedTuple

import devtools

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import type_alias as ta
from icon4py.model.common.grid import geometry as grid_geometry
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.standalone_driver import config as driver_config


log = logging.getLogger(__name__)


class StaticFieldFactories(NamedTuple):
    """
    Factories of static fields for the driver and components.

    Attributes:
        geometry_field_source: grid geometry field factory that stores geometrical properties of a grid
        interpolation_field_source: interpolation field factory that stores pre-computed coefficients for interpolation employed in the model
        metrics_field_source: metrics field factory that stores pre-computed coefficients for numerical operations employed in the model
    """

    geometry_field_source: grid_geometry.GridGeometry
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory
    metrics_field_source: metrics_factory.MetricsFieldsFactory


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
    tracer_advection_diagnostic: advection_states.AdvectionDiagnosticState
    diffusion_diagnostic: diffusion_states.DiffusionDiagnosticState
    prognostics: common_utils.TimeStepPair[prognostics.PrognosticState]
    diagnostic: diagnostics.DiagnosticState


@dataclasses.dataclass
class ModelTimeVariables:
    """
    This class contains driver's run-time time or date variables.
    It tracks the current simulation date, substepping information, and cfl watch mode.

    """

    config: dataclasses.InitVar[driver_config.DriverConfig]

    n_time_steps: int = dataclasses.field(init=False)
    dtime: datetime.timedelta = dataclasses.field(init=False)
    ndyn_substeps_var: int = dataclasses.field(init=False)
    max_ndyn_substeps: int = dataclasses.field(init=False)
    elapsed_time_in_seconds: ta.wpfloat = dataclasses.field(init=False)
    simulation_date: datetime.datetime = dataclasses.field(init=False)
    is_first_step_in_simulation: bool = dataclasses.field(init=False)
    cfl_watch_mode: bool = dataclasses.field(init=False)

    def __post_init__(self, config: driver_config.DriverConfig) -> None:
        self.n_time_steps = int((config.end_date - config.start_date) / config.dtime)
        self.dtime = config.dtime
        self.elapsed_time_in_seconds = ta.wpfloat("0.0")
        self.simulation_date = config.start_date
        self.ndyn_substeps_var = config.ndyn_substeps
        self.max_ndyn_substeps = config.ndyn_substeps + 7
        self.is_first_step_in_simulation = True
        self.cfl_watch_mode = False

        if self.n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    @functools.cached_property
    def dtime_in_seconds(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime.total_seconds())

    @property
    def substep_timestep(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime_in_seconds / self.ndyn_substeps_var)

    def next_simulation_date(self) -> None:
        self.simulation_date += self.dtime
        self.elapsed_time_in_seconds += self.dtime_in_seconds

    def update_ndyn_substeps(self, new_ndyn_substeps: int) -> None:
        self.ndyn_substeps_var = new_ndyn_substeps

    def update_cfl_watch_mode(self, mode: bool) -> None:
        self.cfl_watch_mode = mode

    def reset(self, config: driver_config.DriverConfig) -> None:
        """
        Re-initialize all time-integration-related runtime values from the given config.
        """
        self.__post_init__(config)


class DriverTimers(enum.Enum):
    SOLVE_NH_FIRST_STEP = "solve_nh_first_step"
    SOLVE_NH = "solve_nh"
    DIFFUSION_FIRST_STEP = "diffusion_first_step"
    DIFFUSION = "diffusion"


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
    ) -> None:
        log.info("===== ICON4Py timer report =====")
        table_titles = (
            f"|{'timer name':^30}|"
            f"{'no. of times called':^23}|"
            f"{'mean time (s)':^23}|"
            f"{'std. deviation (s)':^23}|"
            f"{'min time (s)':^23}|"
            f"{'max time (s)':^23}|"
        )
        log.info(table_titles)
        log.info("-" * len(table_titles))
        for timer_name, timer in self.timers.items():
            times = []
            for r in timer.results:
                if not r.finish:
                    r.capture()
                times.append(r.elapsed())
            if len(times) > 0:
                log.info(
                    f"|{timer_name:^30}|"
                    f"{len(times):^23}|"
                    f"{statistics.mean(times):^23.8f}|"
                    f"{statistics.stdev(times) if len(times) > 1 else 0:^23.8f}|"
                    f"{min(times):^23.8f}|"
                    f"{max(times):^23.8f}|"
                )
            else:
                log.info(
                    f"|{timer_name:^30}|"
                    f"{'not started':^23}|"
                    f"{'':^23}|"
                    f"{'':^23}|"
                    f"{'':^23}|"
                    f"{'':^23}|"
                )
