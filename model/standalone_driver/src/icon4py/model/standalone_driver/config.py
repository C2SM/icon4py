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
from typing import Any

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import (
    initial_condition,
    prescribed_tendencies,
    time,
    topography,
    type_alias as ta,
)
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.initial_condition import from_file
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_state
from icon4py.model.common.utils import fortran_config


log = logging.getLogger(__name__)


def absolutetime_from_iconformat(value: str) -> time.AbsoluteTime:
    return time.AbsoluteTime.fromisoformat(value.replace("Z", "+00:00"))


def relativetime_from_iconformat(dtime: float, modeltimestep: str) -> time.RelativeTime:
    return (
        relativetime_from_iso8601(modeltimestep)
        if modeltimestep
        else time.RelativeTime(seconds=dtime)
    )


@dataclasses.dataclass
class ProfilingConfig:
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


def relativetime_from_iso8601(duration: str) -> time.RelativeTime:
    """
    Parse an ISO 8601 duration such as 'PT300S' into a 'time.RelativeTime'.

    Only the components convertible to a fixed duration are supported (weeks,
    days, hours, minutes, seconds).
    """
    match = _ISO8601_DURATION.fullmatch(duration)
    if match is None or not any(match.groups()):
        raise ValueError(f"Invalid ISO 8601 duration: '{duration}'.")
    components = {name: float(value) for name, value in match.groupdict().items() if value}
    return time.RelativeTime(**components)


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
    profiling_options: typing.Annotated[
        ProfilingConfig | None,
        common_conf_opt.ConfigOption(description="Performance profiling options."),
    ]
    dtime: typing.Annotated[
        time.RelativeTime,
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
        time.AbsoluteTime,
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
    start_of_timestepping: typing.Annotated[
        time.AbsoluteTime,
        common_conf_opt.ConfigOption(
            description="Time from when to start or restart (initial run: equivalent to 'start_of_simulation')",
            icon_equivalent=common_conf_opt.IconOption(  # always equal to start_of_simulation when reading from ICON
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
        time.EndOfSimulation,
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
    do_prep_adv: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="No description available yet.",
            icon_equivalent=common_conf_opt.IconOption(
                name="ltransport",
                path=(
                    "model_cfg",
                    "run_nml",
                ),
            ),
        ),
    ] = False  # lprep_adv in fortran
    diffuse_before_time_loop: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="No description available yet.",
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

    def __post_init__(self) -> None:
        if self.start_of_timestepping < self.start_of_simulation:
            raise ValueError(
                f"the time loop cannot start at {self.start_of_timestepping}, before the "
                f"beginning of the simulation ({self.start_of_simulation})."
            )

    @classmethod
    def make_initial(cls, **kwargs: Any) -> DriverConfig:
        kwargs["start_of_timestepping"] = kwargs["start_of_simulation"]
        return cls(**kwargs)

    def __post_init__(self):
        ta.dataclass_scalars_to_wp(self, attributes=["vertical_cfl_threshold"])

    @classmethod
    def from_fortran_dict(
        cls, *, atm_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> DriverConfig:
        # TODO(ricoh): merge the dictionaries outside and put this method in a base class
        return cls.make_initial(
            **dict(
                common_conf_opt.iter_pairs_from_icon(
                    config_cls=cls, icon_config={"master_cfg": master_dict, "model_cfg": atm_dict}
                )
            ),
            **overrides,
        )


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    geometry: GeometryConfig
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    topography: topography.TopographyConfig
    initial_condition: initial_condition.InitialConditionConfig
    prescribed_tendencies: prescribed_tendencies.PrescribedTendenciesConfig
    driver: DriverConfig
    nonhydrostatic: solve_nh.NonHydrostaticConfig | None = None
    diffusion: diffusion.DiffusionConfig | None = None
    tracer_config: tracer_state.TracerConfig | None = None
    tracer_advection: tracer_advection.AdvectionConfig | None = None
    graupel: graupel.SingleMomentSixClassIconGraupelConfig | None = None

    def __post_init__(self) -> None:
        # The file-based initial condition needs the clock of the driver to know which
        # savepoint to read: the initial state, or the state of a later time step when
        # restarting. 'with_overrides' rebuilds the config, so the two stay in sync.
        initial_condition_config = self.initial_condition.config
        if isinstance(initial_condition_config, from_file.FromFileConfig):
            initial_condition_config.start_of_simulation = self.driver.start_of_simulation
            initial_condition_config.start_of_timestepping = self.driver.start_of_timestepping
            initial_condition_config.dtime = self.driver.dtime

        if self.driver.diffuse_before_time_loop and not (
            self.nonhydrostatic is not None
            and self.diffusion is not None
            and self.diffusion.apply_to_horizontal_wind
        ):
            object.__setattr__(
                self, "driver", dataclasses.replace(self.driver, diffuse_before_time_loop=False)
            )

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, dict):
                replacements[key] = dataclasses.replace(current, **value)
            else:
                replacements[key] = value
        return dataclasses.replace(self, **replacements)


def read_experiment_config_from_fortran(
    config_file_path: pathlib.Path,
    *,
    enable_profiling: bool = False,
    enable_statistics_output: bool = False,
) -> ExperimentConfig:
    """Assemble an :class:`ExperimentConfig` from a directory of serialized Fortran namelists."""

    with (config_file_path / fortran_config.ATM_DICT_FNAME).open() as f:
        atm_dict = json.load(f)
    with (config_file_path / fortran_config.MASTER_DICT_FNAME).open() as f:
        master_dict = json.load(f)
    with (config_file_path / fortran_config.INPUT_DICT_FNAME).open() as f:
        input_dict = json.load(f)

    geometry_cfg = GeometryConfig(use_analytical_means=True)

    metrics_cfg = metrics_factory.MetricsConfig.from_fortran_dict(atm_dict)

    interpolation_cfg = interpolation_factory.InterpolationConfig.from_fortran_dict(atm_dict)

    vertical_grid_cfg = v_grid.VerticalGridConfig.from_fortran_dict(atm_dict)

    topography_cfg = topography.TopographyConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    nonhydro_cfg = solve_nh.NonHydrostaticConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_cfg.max_nudging_coefficient,
    )

    diffusion_cfg = diffusion.DiffusionConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_cfg.max_nudging_coefficient,
    )

    do_tracer_advection = not (
        "exclaim_ch_r04b09_dsl" in config_file_path.name
        or "exclaim_ape_R02B04" in config_file_path.name
    )
    # The experiments above were run in fortran with a tracer advection scheme
    # that has not been ported to ICON4Py and can not be used for testing.
    # TODO (jcanton): this isn't the right place to keep a special case
    # handling. Either fix these experiments or move the special case handling.
    tracer_advection_cfg = (
        tracer_advection.AdvectionConfig.from_fortran_dict(atm_dict)
        if do_tracer_advection
        else None
    )
    ntracer = (
        fortran_config.list_to_value(atm_dict["run_nml"]["ntracer"]) if do_tracer_advection else 0
    )
    tracer_cfg = tracer_state.TracerConfig.from_ntracer(ntracer)

    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    # If these two namelists are missing it means that the experiment was run
    # without microphysics and we have to skip parsing the graupel config which
    # relies on some of these parameters.
    graupel_cfg = (
        graupel.SingleMomentSixClassIconGraupelConfig.from_fortran_dict(atm_dict)
        if do_physics
        else None
    )

    profiling_stats = ProfilingConfig() if enable_profiling else None
    driver_cfg = DriverConfig.from_fortran_dict(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_options=profiling_stats,
        enable_statistics_logging=enable_statistics_output,
    )

    # the file-based initial condition needs the clock of the driver to know which
    # savepoint to read: the initial state, or a later one when restarting
    initial_condition_cfg = initial_condition.InitialConditionConfig.from_fortran_dict(
        atm_dict=atm_dict,
        input_dict=input_dict,
        data_path=config_file_path,
        start_of_simulation=driver_cfg.start_of_simulation,
        start_of_timestepping=driver_cfg.start_of_timestepping,
        dtime=driver_cfg.dtime,
    )

    if not do_tracer_advection and isinstance(
        initial_condition_cfg.config, from_file.FromFileConfig
    ):
        initial_condition_cfg = dataclasses.replace(
            initial_condition_cfg,
            config=dataclasses.replace(initial_condition_cfg.config, ntracer=0),
        )

    return ExperimentConfig(
        geometry=geometry_cfg,
        metrics=metrics_cfg,
        interpolation=interpolation_cfg,
        vertical_grid=vertical_grid_cfg,
        topography=topography_cfg,
        nonhydrostatic=nonhydro_cfg,
        diffusion=diffusion_cfg,
        tracer_config=tracer_cfg,
        tracer_advection=tracer_advection_cfg,
        graupel=graupel_cfg,
        initial_condition=initial_condition_cfg,
        prescribed_tendencies=prescribed_tendencies.PrescribedTendenciesConfig.from_fortran_dict(
            atm_dict=atm_dict, data_path=config_file_path
        ),
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
            current_time = time.AbsoluteTime.now()
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
