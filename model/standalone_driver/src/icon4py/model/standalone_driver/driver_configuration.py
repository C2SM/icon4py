# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import datetime
import logging
import os
import pathlib

import gt4py.next.typing as gtx_typing
from gt4py.next import metrics as gtx_metrics

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import model_backends, type_alias as ta
from icon4py.model.common.grid import vertical as v_grid


log = logging.getLogger(__name__)


@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    experiment_name: str
    backend_name: dataclasses.InitVar[str]
    backend: gtx_typing.Backend = dataclasses.field(init=False)
    grid_file_path: pathlib.Path
    configuration_file_path: pathlib.Path
    output_path: pathlib.Path
    profiling_stats: ProfilingStats | None

    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)

    apply_extra_second_order_divdamp: bool = False

    vertical_cfl_threshold: ta.wpfloat = 0.85

    ndyn_substeps: int = 5

    enable_statistics_output: bool = False

    def __post_init__(self, backend_name):
        if backend_name not in model_backends.BACKENDS:
            raise ValueError(
                f"Invalid driver backend: {backend_name}. \n"
                f"Available backends are {', '.join([f'{k}' for k in model_backends.BACKENDS])}"
            )
        backend_factory = model_backends.BACKENDS[backend_name]["backend_factory"]
        device = model_backends.BACKENDS[backend_name]["device"]
        object.__setattr__(self, "backend", backend_factory(device))

        for name in ["grid_file_path", "configuration_file_path", "output_path"]:
            path = (
                pathlib.Path(os.path.expandvars(str(getattr(self, name)))).expanduser().absolute()
            )
            object.__setattr__(self, name, path)
            if not path.exists():
                log.warning(f"The path for {name} does not exist: {path}")
            # elif not path.is_file():
            #     raise ValueError(f"{name} must be a file: {path}")


def  read_config(
    configuration_file_path: pathlib.Path,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    backend_name: str,
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

    driver_run_config = DriverConfig(
        experiment_name="Jablonowski_Williamson",
        backend_name=backend_name,
        output_path=output_path,
        configuration_file_path=configuration_file_path,
        grid_file_path=grid_file_path,
        dtime=datetime.timedelta(seconds=300.0),
        end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
        apply_extra_second_order_divdamp=False,
        ndyn_substeps=5,
        vertical_cfl_threshold=ta.wpfloat("0.85"),
        enable_statistics_output=True,
        profiling_stats=profiling_stats,
    )

    return (
        driver_run_config,
        vertical_grid_config,
        diffusion_config,
        nonhydro_config,
    )
