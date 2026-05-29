# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib

from icon4py.model.atmosphere.advection import advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.testing import definitions


def read_config(
    config_file_path: pathlib.Path,
    enable_profiling: bool = False,
) -> tuple[
    driver_config.DriverConfig,
    v_grid.VerticalGridConfig,
    diffusion.DiffusionConfig,
    advection.AdvectionConfig,
    solve_nh.NonHydrostaticConfig,
    interpolation_factory.InterpolationConfig,
    metrics_factory.MetricsConfig,
]:
    with (config_file_path / f"{definitions.NAMELIST_ATM_FNAME}.json").open() as f:
        atmo_dict = json.load(f)
    with (config_file_path / f"{definitions.NAMELIST_MASTER_FNAME}.json").open() as f:
        master_dict = json.load(f)

    profiling_stats = driver_config.ProfilingStats() if enable_profiling else None
    interpolation_config = interpolation_factory.InterpolationConfig.from_fortran_dict(atmo_dict)
    metrics_config = metrics_factory.MetricsConfig.from_fortran_dict(atmo_dict)

    driver_cfg = driver_config.DriverConfig.from_fortran_dict(
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

