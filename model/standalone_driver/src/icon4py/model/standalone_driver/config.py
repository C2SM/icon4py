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
import json
import logging
import pathlib
from typing import Any, TypeAlias

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import topography, type_alias as ta
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.utils import fortran_config
from icon4py.model.standalone_driver import initial_condition


log = logging.getLogger(__name__)


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
    ntracer: int = 0

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
    # NOTE: This has a duplicate in testing/definitions.py to avoid circular imports.
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    topography: topography.TopographyConfig
    initial_condition: initial_condition.InitialConditionConfig
    driver: DriverConfig
    nonhydrostatic: solve_nh.NonHydrostaticConfig | None = None
    diffusion: diffusion.DiffusionConfig | None = None
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

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, dict):
                replacements[key] = dataclasses.replace(current, **value)
            else:
                replacements[key] = value
        return dataclasses.replace(self, **replacements)


def read_config(
    config_file_path: pathlib.Path,
    enable_profiling: bool = False,
) -> ExperimentConfig:
    # NOTE: This has a duplicate in testing/datatest_utils.py to avoid circular imports.

    with (config_file_path / fortran_config.ATM_DICT_FNAME).open() as f:
        atm_dict = json.load(f)
    with (config_file_path / fortran_config.MASTER_DICT_FNAME).open() as f:
        master_dict = json.load(f)
    with (config_file_path / fortran_config.INPUT_DICT_FNAME).open() as f:
        input_dict = json.load(f)

    metrics_config = metrics_factory.MetricsConfig.from_fortran_dict(atm_dict)

    interpolation_config = interpolation_factory.InterpolationConfig.from_fortran_dict(atm_dict)

    vertical_grid_config = v_grid.VerticalGridConfig.from_fortran_dict(atm_dict)

    topography_config = topography.TopographyConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    nonhydro_config = solve_nh.NonHydrostaticConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )

    diffusion_config = diffusion.DiffusionConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )

    do_tracer_advection = not (
        "exclaim_ch_r04b09_dsl" in config_file_path.name
        or "exclaim_ape_R02B04" in config_file_path.name
    )
    # The experiments above were run in fortran with a tracer advection scheme
    # that has not been ported to ICON4Py and can not be used for testing.
    tracer_advection_config = (
        tracer_advection.AdvectionConfig.from_fortran_dict(atm_dict)
        if do_tracer_advection
        else None
    )

    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    # If these two namelists are missing it means that the experiment was run
    # without microphysics and we have to skip parsing the graupel config which
    # relies on some of these parameters.
    graupel_config = (
        graupel.SingleMomentSixClassIconGraupelConfig.from_fortran_dict(atm_dict)
        if do_physics
        else None
    )

    initial_condition_config = initial_condition.InitialConditionConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    profiling_stats = ProfilingStats() if enable_profiling else None
    driver_cfg = DriverConfig.from_fortran_dict(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_stats=profiling_stats,
        ntracer=fortran_config.list_to_value(atm_dict["run_nml"]["ntracer"])
        if do_tracer_advection
        else 0,
    )

    return ExperimentConfig(
        metrics=metrics_config,
        interpolation=interpolation_config,
        vertical_grid=vertical_grid_config,
        topography=topography_config,
        nonhydrostatic=nonhydro_config,
        diffusion=diffusion_config,
        tracer_advection=tracer_advection_config,
        graupel=graupel_config,
        initial_condition=initial_condition_config,
        driver=driver_cfg,
    )


def prepare_output_directory(
    config_output_path: pathlib.Path,
    cli_output_path: pathlib.Path | None,
    process_props: Any | None = None,
) -> pathlib.Path:
    output_path = cli_output_path if cli_output_path is not None else config_output_path

    is_rank_zero = process_props is None or process_props.rank == 0

    if is_rank_zero:
        if output_path.exists():
            current_time = datetime.datetime.now()
            log.warning(f"output path {output_path} already exists, a time stamp will be added")
            output_path = (
                output_path.parent
                / f"{output_path.name}_{datetime.date.today()}_{current_time.hour}h_{current_time.minute}m_{current_time.second}s"
            )
        output_path.mkdir(parents=True, exist_ok=False)

    if process_props is not None and process_props.comm_size > 1:
        output_path = pathlib.Path(
            process_props.comm.bcast(str(output_path) if process_props.rank == 0 else None, root=0)
        )
        process_props.comm.Barrier()

    return output_path
