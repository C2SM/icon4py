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
import re
import typing
from typing import Any, TypeAlias

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import topography, type_alias as ta
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_state
from icon4py.model.common.utils import fortran_config
from icon4py.model.standalone_driver import initial_condition
from icon4py.model.standalone_driver.initial_condition import from_file as from_file_ic


log = logging.getLogger(__name__)


RelativeTime: TypeAlias = datetime.timedelta
AbsoluteTime: TypeAlias = datetime.datetime
NumTimeSteps: TypeAlias = int
EndOfSimulation: TypeAlias = RelativeTime | AbsoluteTime | NumTimeSteps


def absolutetime_from_iconformat(value: str) -> AbsoluteTime:
    return AbsoluteTime.fromisoformat(value.replace("Z", "+00:00"))


def relativetime_from_iconformat(dtime: float, modeltimestep: str) -> RelativeTime:
    return (
        relativetime_from_iso8601(modeltimestep) if modeltimestep else RelativeTime(seconds=dtime)
    )


@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


# ISO 8601 duration, restricted to the fixed-length components (weeks, days,
# hours, minutes, seconds). Years and months are intentionally not matched since
# their length is not fixed, and this is currently only used for dtime.
_ISO8601_DURATION = re.compile(
    r"P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?=\d)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)


def relativetime_from_iso8601(duration: str) -> RelativeTime:
    """
    Parse an ISO 8601 duration such as 'PT300S' into a 'RelativeTime'.

    Only the components convertible to a fixed duration are supported (weeks,
    days, hours, minutes, seconds).
    """
    match = _ISO8601_DURATION.fullmatch(duration)
    if match is None or not any(match.groups()):
        raise ValueError(f"Invalid ISO 8601 duration: '{duration}'.")
    components = {name: float(value) for name, value in match.groupdict().items() if value}
    return RelativeTime(**components)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DriverConfig:
    """
    Standalone driver configuration.

    Default values should correspond to default values in ICON.
    """

    experiment_name: typing.Annotated[
        str,
        common_conf_opt.ConfigOption(
            description="Name of the experiment",
            icon_equivalent=common_conf_opt.IconOption(
                name="model_namelist_filename",
                path=(
                    "master_cfg",
                    "master_model_nml",
                ),
                converter=lambda model_namelist_filename: model_namelist_filename.removeprefix(
                    "NAMELIST_"
                ).removesuffix("_sb_atm"),
            ),
        ),
    ]
    profiling_stats: typing.Annotated[
        ProfilingStats | None,
        common_conf_opt.ConfigOption(
            description=""  # TODO(ricoh): c35 -- Add a description
        ),
    ]
    dtime: typing.Annotated[
        RelativeTime,
        common_conf_opt.ConfigOption(
            description="Time step duration.",
            icon_equivalent=common_conf_opt.IconMultiOption(
                options=[
                    common_conf_opt.IconOption(
                        name="dtime",
                        path=(
                            "model_cfg",
                            "run_nml",
                        ),
                    ),
                    common_conf_opt.IconOption(
                        name="modeltimestep",
                        path=(
                            "model_cfg",
                            "run_nml",
                        ),
                        converter=str.strip,
                    ),
                ],
                converter=relativetime_from_iconformat,
            ),
        ),
    ]
    start_of_simulation: typing.Annotated[
        AbsoluteTime,
        common_conf_opt.ConfigOption(
            description="Start date and time of a simulation.",
            icon_equivalent=common_conf_opt.IconOption(
                name="experimentstartdate",
                path=(
                    "master_cfg",
                    "master_time_control_nml",
                ),
                converter=absolutetime_from_iconformat,
            ),
        ),
    ]
    end_of_simulation: typing.Annotated[
        EndOfSimulation,
        common_conf_opt.ConfigOption(
            description="End date and time of a simulation.",
            icon_equivalent=common_conf_opt.IconOption(
                name="experimentstopdate",
                path=(
                    "master_cfg",
                    "master_time_control_nml",
                ),
                converter=absolutetime_from_iconformat,
            ),
        ),
    ]
    output_path: typing.Annotated[
        pathlib.Path,
        common_conf_opt.ConfigOption(
            description="Output directory path, relative to the working directory.",
            icon_equivalent=None,
        ),
    ] = dataclasses.field(default_factory=lambda: pathlib.Path("./output"))
    apply_extra_second_order_divdamp: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Whether or not to apply additional second order divergence damping. "
                "Not a namelist variable, coded as follows in mo_nh_stepping.f90: "
                "# IF (elapsed_time_global <= 7200._wp+0.5_wp*dtime .AND. .NOT. ltestcase)"
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="ltestcase",
                path=(
                    "model_cfg",
                    "run_nml",
                ),
                converter=lambda value: not value,
            ),
        ),
    ] = False
    vertical_cfl_threshold: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Threshold for vertical advection CFL number at which the adaptive time step reduction "
                "(increase of ndyn_substeps w.r.t. the fixed fast-physics time step) is triggered."
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="vcfl_threshold",
                path=(
                    "model_cfg",
                    "nonhydrostatic_nml",
                ),
            ),
        ),
    ] = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Number of dynamics substeps per time step.",
            icon_equivalent=common_conf_opt.IconOption(
                "ndyn_substeps",
                (
                    "model_cfg",
                    "nonhydrostatic_nml",
                ),
            ),
        ),
    ] = 5
    enable_statistics_logging: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="Compute and log variable statistics.",
            icon_equivalent=None,
        ),
    ] = False
    enable_output: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Enable output to file. For now this is only documented in "
                "'icon4py.model.standalone_driver.driver_io'."
            ),
            icon_equivalent=None,
        ),
    ] = False

    @classmethod
    def from_fortran_dict(
        cls, *, atm_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> DriverConfig:
        # TODO(ricoh): merge the dictionaries outside and put this method in a base class
        return common_conf_opt.construct_config_from_icon(
            config_cls=cls,
            icon_config={"master_cfg": master_dict, "model_cfg": atm_dict},
            **overrides,
        )


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    # NOTE: This has a duplicate in testing/definitions.py to avoid circular imports.
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    topography: topography.TopographyConfig
    geometry: GeometryConfig
    initial_condition: initial_condition.InitialConditionConfig
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
    # TODO (jcanton): this isn't the right place to keep a special case
    # handling. Either fix these experiments or move the special case handling.
    tracer_advection_config = (
        tracer_advection.AdvectionConfig.from_fortran_dict(atm_dict)
        if do_tracer_advection
        else None
    )
    ntracer = (
        fortran_config.list_to_value(atm_dict["run_nml"]["ntracer"]) if do_tracer_advection else 0
    )
    tracer_config = tracer_state.TracerConfig.from_ntracer(ntracer)

    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    # If these two namelists are missing it means that the experiment was run
    # without microphysics and we have to skip parsing the graupel config which
    # relies on some of these parameters.
    graupel_config = (
        graupel.SingleMomentSixClassIconGraupelConfig.from_fortran_dict(atm_dict)
        if do_physics
        else None
    )

    geometry_config = GeometryConfig(use_analytical_means=True)

    initial_condition_config = initial_condition.InitialConditionConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    if not do_tracer_advection and isinstance(
        initial_condition_config.config, from_file_ic.FromFileConfig
    ):
        initial_condition_config = dataclasses.replace(
            initial_condition_config,
            config=dataclasses.replace(initial_condition_config.config, ntracer=0),
        )

    profiling_stats = ProfilingStats() if enable_profiling else None
    driver_cfg = DriverConfig.from_fortran_dict(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_stats=profiling_stats,
    )

    return ExperimentConfig(
        metrics=metrics_config,
        interpolation=interpolation_config,
        vertical_grid=vertical_grid_config,
        topography=topography_config,
        geometry=geometry_config,
        nonhydrostatic=nonhydro_config,
        diffusion=diffusion_config,
        tracer_config=tracer_config,
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
            current_time = AbsoluteTime.now()
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
