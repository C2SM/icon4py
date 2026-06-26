# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import json
import pathlib

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import topography
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.initial_condition import config as initial_condition_config
from icon4py.model.common.initial_condition.from_file import FromFileConfig
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_state
from icon4py.model.common.utils import fortran_config
from icon4py.model.experiment_config.config import DriverConfig, ExperimentConfig, ProfilingStats


def read_experiment_config(
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

    initial_condition_cfg = initial_condition_config.InitialConditionConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    if not do_tracer_advection and isinstance(initial_condition_cfg.config, FromFileConfig):
        initial_condition_cfg = dataclasses.replace(
            initial_condition_cfg,
            config=dataclasses.replace(initial_condition_cfg.config, ntracer=0),
        )

    profiling_stats = ProfilingStats() if enable_profiling else None
    driver_cfg = DriverConfig.from_fortran_dict(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_stats=profiling_stats,
        enable_statistics_output=enable_statistics_output,
    )

    return ExperimentConfig(
        metrics=metrics_config,
        interpolation=interpolation_config,
        vertical_grid=vertical_grid_config,
        topography=topography_config,
        nonhydrostatic=nonhydro_config,
        diffusion=diffusion_config,
        tracer_config=tracer_config,
        tracer_advection=tracer_advection_config,
        graupel=graupel_config,
        initial_condition=initial_condition_cfg,
        driver=driver_cfg,
    )
