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
from typing import Any

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.advection import advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import type_alias as ta
from icon4py.model.common.decomposition import mpi_decomposition as mpi_decomp
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.utils import fortran_config


log = logging.getLogger(__name__)


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
    dtime: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    output_path: pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path("./output"))
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False
    ntracer: int = 0

    @classmethod
    def from_fortran_dict(
        cls, atmo_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> DriverConfig:
        nonhydrostatic_nml = atmo_dict["nonhydrostatic_nml"]
        run_nml = atmo_dict["run_nml"]
        master_time_control_nml = master_dict["master_time_control_nml"]
        dtime = run_nml["dtime"]
        start_date_str = master_time_control_nml["experimentstartdate"]
        end_date_str = master_time_control_nml["experimentstopdate"]
        return cls(
            dtime=datetime.timedelta(seconds=dtime),
            start_date=datetime.datetime.fromisoformat(start_date_str.replace("Z", "+00:00")),
            end_date=datetime.datetime.fromisoformat(end_date_str.replace("Z", "+00:00")),
            apply_extra_second_order_divdamp=nonhydrostatic_nml["lextra_diffu"],
            vertical_cfl_threshold=ta.wpfloat(str(nonhydrostatic_nml["vcfl_threshold"])),
            ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
            ntracer=fortran_config.list_to_value(run_nml["ntracer"]),
            **overrides,
        )


def read_config(
    config_file_path: pathlib.Path,
    enable_profiling: bool = False,
) -> tuple[
    DriverConfig,
    v_grid.VerticalGridConfig,
    diffusion.DiffusionConfig,
    advection.AdvectionConfig,
    solve_nh.NonHydrostaticConfig,
    interpolation_factory.InterpolationConfig,
    metrics_factory.MetricsConfig,
]:
    with (config_file_path / f"{fortran_config.NAMELIST_ATM_FNAME}.json").open() as f:
        atmo_dict = json.load(f)
    with (config_file_path / f"{fortran_config.NAMELIST_MASTER_FNAME}.json").open() as f:
        master_dict = json.load(f)

    profiling_stats = ProfilingStats() if enable_profiling else None

    interpolation_config = interpolation_factory.InterpolationConfig.from_fortran_dict(atmo_dict)
    assert interpolation_config.max_nudging_coefficient is not None

    metrics_config = metrics_factory.MetricsConfig.from_fortran_dict(atmo_dict)

    driver_cfg = DriverConfig.from_fortran_dict(
        atmo_dict,
        master_dict,
        experiment_name="standalone",
        profiling_stats=profiling_stats,
    )
    vertical_grid_config = v_grid.VerticalGridConfig.from_fortran_dict(atmo_dict)
    diffusion_config = diffusion.DiffusionConfig.from_fortran_dict(
        atmo_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )
    advection_config = advection.AdvectionConfig.from_fortran_dict(atmo_dict)
    nonhydro_config = solve_nh.NonHydrostaticConfig.from_fortran_dict(
        atmo_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )

    return (
        driver_cfg,
        vertical_grid_config,
        diffusion_config,
        advection_config,
        nonhydro_config,
        interpolation_config,
        metrics_config,
    )


def prepare_output_directory(
    output_path: pathlib.Path,
    process_props: Any | None = None,
) -> pathlib.Path:
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

    if process_props is not None:
        if mpi_decomp.mpi4py is not None and mpi_decomp.mpi4py.MPI.COMM_WORLD.Get_size() > 1:
            comm = mpi_decomp.mpi4py.MPI.COMM_WORLD
            output_path = pathlib.Path(
                comm.bcast(str(output_path) if process_props.rank == 0 else None, root=0)
            )
            comm.Barrier()

    return output_path
