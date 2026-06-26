# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import datetime
import pathlib
from typing import TYPE_CHECKING, Any, TypeAlias

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.common import type_alias as ta


if TYPE_CHECKING:
    from icon4py.model.atmosphere.advection import advection as tracer_advection
    from icon4py.model.atmosphere.diffusion import diffusion
    from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
    from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
        single_moment_six_class_gscp_graupel as graupel,
    )
    from icon4py.model.common import topography
    from icon4py.model.common.grid import vertical as v_grid
    from icon4py.model.common.initial_condition.config import InitialConditionConfig
    from icon4py.model.common.interpolation import interpolation_factory
    from icon4py.model.common.metrics import metrics_factory
    from icon4py.model.common.states import tracer_state


RelativeTime: TypeAlias = datetime.timedelta
AbsoluteTime: TypeAlias = datetime.datetime
NumTimeSteps: TypeAlias = int
EndOfSimulation: TypeAlias = RelativeTime | AbsoluteTime | NumTimeSteps


@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Standalone driver configuration.

    Default values should correspond to default values in ICON.
    """

    experiment_name: str
    profiling_stats: ProfilingStats | None
    dtime: RelativeTime
    start_of_simulation: AbsoluteTime
    end_of_simulation: EndOfSimulation
    output_path: pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path("./output"))
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False
    enable_output: bool = False

    @classmethod
    def from_fortran_dict(
        cls, *, atm_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> DriverConfig:
        nonhydrostatic_nml = atm_dict["nonhydrostatic_nml"]
        run_nml = atm_dict["run_nml"]
        master_time_control_nml = master_dict["master_time_control_nml"]
        master_model_nml = master_dict["master_model_nml"]
        dtime = run_nml["dtime"]
        start_datetime_str = master_time_control_nml["experimentstartdate"]
        end_datetime_str = master_time_control_nml["experimentstopdate"]
        return cls(
            experiment_name=master_model_nml["model_namelist_filename"]
            .removeprefix("NAMELIST_")
            .removesuffix("_sb_atm"),
            dtime=datetime.timedelta(seconds=dtime),
            start_of_simulation=datetime.datetime.fromisoformat(
                start_datetime_str.replace("Z", "+00:00")
            ),
            end_of_simulation=datetime.datetime.fromisoformat(
                end_datetime_str.replace("Z", "+00:00")
            ),
            # apply_extra_second_order_divdamp does not have a namelist
            # variable in fortran. It is coded as follows in mo_nh_stepping.f90:
            # IF (elapsed_time_global <= 7200._wp+0.5_wp*dtime .AND. .NOT. ltestcase)
            apply_extra_second_order_divdamp=not run_nml.get("ltestcase", False),
            vertical_cfl_threshold=ta.wpfloat(str(nonhydrostatic_nml["vcfl_threshold"])),
            ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
            **overrides,
        )


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    topography: topography.TopographyConfig
    initial_condition: InitialConditionConfig
    driver: DriverConfig
    nonhydrostatic: solve_nh.NonHydrostaticConfig | None = None
    diffusion: diffusion.DiffusionConfig | None = None
    tracer_config: tracer_state.TracerConfig | None = None
    tracer_advection: tracer_advection.AdvectionConfig | None = None
    graupel: graupel.SingleMomentSixClassIconGraupelConfig | None = None

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, dict):
                replacements[key] = dataclasses.replace(current, **value)
            else:
                replacements[key] = value
        return dataclasses.replace(self, **replacements)
