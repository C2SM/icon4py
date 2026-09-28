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
import logging
import pathlib
import typing
from typing import Any

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys import config as muphys_config
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import (
    backend_configuration as backend_cfg,
    initial_condition,
    prescribed_tendencies,
    time,
    topography,
    type_alias as ta,
)
from icon4py.model.common.config import config_io, options as common_conf_opt
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.initial_condition import from_file
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.io import io as common_io
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_states
from icon4py.model.common.utils.time_utils import relativetime_from_iso8601


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
        ),
    ]
    start_of_simulation: typing.Annotated[
        time.AbsoluteTime,
        common_conf_opt.ConfigOption(
            description="Start date and time of a simulation.",
        ),
    ]
    start_of_timestepping: typing.Annotated[
        time.AbsoluteTime,
        common_conf_opt.ConfigOption(
            description="Time from when to start or restart (initial run: equivalent to 'start_of_simulation')",
        ),
    ]
    end_of_simulation: typing.Annotated[
        time.EndOfSimulation,
        common_conf_opt.ConfigOption(
            description="End date and time of a simulation.",
        ),
    ]
    output_path: typing.Annotated[
        pathlib.Path,
        common_conf_opt.ConfigOption(
            description="Output directory path, relative to the working directory.",
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
        ),
    ] = False
    diffuse_before_time_loop: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="No description available yet.",
        ),
    ] = False
    vertical_cfl_threshold: typing.Annotated[
        ta.wpfloat,
        common_conf_opt.ConfigOption(
            description=(
                "Threshold for vertical advection CFL number at which the adaptive time step reduction "
                "(increase of ndyn_substeps w.r.t. the fixed fast-physics time step) is triggered."
            ),
        ),
    ] = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Number of dynamics substeps per time step.",
        ),
    ] = 5
    enable_statistics_logging: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="Compute and log variable statistics.",
        ),
    ] = False
    enable_output: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Enable output to file. For now this is only documented in "
                "'icon4py.model.driver.driver_io'."
            ),
        ),
    ] = False
    backend_config: typing.Annotated[
        backend_cfg.BackendConfig | None,
        common_conf_opt.ConfigOption(
            description=(
                "Backend configuration options, which affect performance but not "
                "the scientific outcome. `None` falls back to environment variables, "
                "if set, otherwise the default configuration is used."
            ),
        ),
    ] = dataclasses.field(default_factory=backend_cfg.backend_config_from_env)
    output_backend: typing.Annotated[
        common_io.OutputBackend,
        common_conf_opt.ConfigOption(
            description="File format of the output field groups ('netcdf' or 'zarr').",
        ),
    ] = common_io.OutputBackend.ZARR
    output_mode: typing.Annotated[
        common_io.OutputMode,
        common_conf_opt.ConfigOption(
            description=(
                "Write strategy of distributed runs ('gather' or 'distributed', see "
                "'icon4py.model.common.io.OutputMode'); 'distributed' netCDF needs an "
                "MPI-parallel netCDF4 installation in multi-rank runs (see 'Parallel "
                "netCDF' in 'icon4py.model.common.io')."
            ),
        ),
    ] = common_io.OutputMode.DISTRIBUTED

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentConfig(config_io.ConfigWithShared):
    geometry: GeometryConfig
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    initial_condition: initial_condition.IC_CONFIG
    topography: topography.TOPO_CONFIG
    prescribed_tendencies: prescribed_tendencies.PrescribedTendenciesConfig
    driver: DriverConfig
    nonhydrostatic: solve_nh.NonHydrostaticConfig | None = None
    diffusion: diffusion.DiffusionConfig | None = None
    tracer_config: tracer_states.TracerConfig | None = None
    tracer_advection: tracer_advection.AdvectionConfig | None = None
    graupel: graupel.SingleMomentSixClassIconGraupelConfig | None = None
    muphys: muphys_config.MuphysConfig | None = None
    #: Read from the AES vertical-diffusion namelist; the driver does not run the
    #: granule yet (icon4py#1360), so it is carried but unused.
    tmx: tmx_config.TmxConfig | None = None

    def __post_init__(self) -> None:
        # The file-based initial condition needs the clock of the driver to know which
        # savepoint to read: the initial state, or the state of a later time step when
        # restarting. 'with_overrides' rebuilds the config, so the two stay in sync.
        initial_condition_config = self.initial_condition
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


def read_experiment_config_from_yaml(config_file_path: pathlib.Path) -> ExperimentConfig:
    """Read an :class:`ExperimentConfig`, resolving relative ``data_path`` entries against the file's directory."""
    config = config_io.read_yaml_str(config_file_path.read_text(), ExperimentConfig)
    root = config_file_path.resolve().parent
    overrides: dict[str, Any] = {}
    for field in dataclasses.fields(config):
        data_path = getattr(getattr(config, field.name), "data_path", None)
        if isinstance(data_path, pathlib.Path) and not data_path.is_absolute():
            overrides[field.name] = {"data_path": root / data_path}
    return config.with_overrides(**overrides)


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
