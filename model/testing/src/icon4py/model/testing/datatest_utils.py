# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import json
import pathlib
import urllib.parse

import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.advection import advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.utils import fortran_config
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.testing import data_handling, definitions, serialbox


def get_process_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_process_properties(run_instance)


def get_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription,
) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment_description.name}_v{experiment_description.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment_description)}"


def get_experiment_archive_filename(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment_description, comm_size)}.tar.gz"


def get_experiment_archive_url(root_url: str, filepath: str) -> str:
    """Build a download URL for experiment archive from root URL."""
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_grid_archive_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.tar.gz"


def get_grid_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.nc"


def get_grid_filepath(grid: definitions.GridDescription) -> pathlib.Path:
    return definitions.grids_path().joinpath(grid.name, get_grid_filename(grid))


def get_grid_archive_url(root_url: str, grid: definitions.GridDescription) -> str:
    """Build a download URL for a grid archive from root URL."""
    filepath = f"{definitions.GRID_DATA_DIR}/{get_grid_archive_filename(grid)}"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_muphys_archive_url(root_url: str, experiment_type: str, experiment_name: str) -> str:
    """Build a download URL for a muphys archive from root URL."""
    filepath = f"{definitions.MUPHYS_DATA_DIR}/{experiment_type}/{experiment_name}.tar.gz"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment_description,
        process_props.comm_size,
    )
    return definitions.serialized_data_path().joinpath(
        experiment_dir, definitions.SERIALIZED_DATA_SUBDIR
    )


def create_icon_serial_data_provider(
    datapath: pathlib.Path,
    rank: int,
    backend: gtx_typing.Backend | None,
) -> serialbox.IconSerialDataProvider:
    return serialbox.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=rank,
        do_print=True,
    )


def download_experiment(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> None:
    """Download data and config for an experiment--if not already present."""
    comm_size = processor_props.comm_size
    root_url = definitions.TESTDATA_ROOT_URL
    archive_filename = get_experiment_archive_filename(experiment_description, comm_size)
    archive_path = definitions.EXPERIMENT_DATA_DIR + "/" + archive_filename
    uri = get_experiment_archive_url(root_url, archive_path)
    destination_path = get_datapath_for_experiment(experiment_description, processor_props)
    data_handling.download_test_data(destination_path.parent, uri)


@functools.cache
def _read_namelist_json(json_file_path: pathlib.Path) -> dict:
    """
    Read and cache the namelist JSON file.

    Args:
        json_file_path: Path to the NAMELIST_ICON_output_atm.json file

    Returns:
        Dictionary containing the parsed JSON data
    """
    with json_file_path.open() as f:
        return json.load(f)


def create_experiment_configuration(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> definitions.ExperimentConfig:
    """
    Create configuration objects from the experiment's namelist JSON files.

    This function reads the NAMELIST_ICON_output_atm.json and icon_master.namelist.json
    files that come with the serialized experiment data and constructs configuration
    objects using from_fortran_dict classmethods.

    Args:
        experiment_description: The experiment definition
        processor_props: Processor properties containing comm_size

    Returns:
        ExperimentConfig object containing the configuration for the experiment
    """

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment_description,
        processor_props.comm_size,
    )
    config_path = definitions.serialized_data_path() / experiment_dir

    atmo_dict = _read_namelist_json(config_path / f"{fortran_config.NAMELIST_ATM_FNAME}.json")
    master_dict = _read_namelist_json(config_path / f"{fortran_config.NAMELIST_MASTER_FNAME}.json")

    interpolation_config = interpolation_factory.InterpolationConfig.from_fortran_dict(atmo_dict)
    assert interpolation_config.max_nudging_coefficient is not None

    metrics_config = metrics_factory.MetricsConfig.from_fortran_dict(atmo_dict)

    driver_cfg = driver_config.DriverConfig.from_fortran_dict(
        atmo_dict,
        master_dict,
        experiment_name=experiment_description.name,
        profiling_stats=None,
        enable_statistics_output=False,
    )

    vertical_grid_config = v_grid.VerticalGridConfig.from_fortran_dict(atmo_dict)

    nonhydro_config = solve_nh.NonHydrostaticConfig.from_fortran_dict(
        atmo_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )

    diffusion_config = diffusion.DiffusionConfig.from_fortran_dict(
        atmo_dict,
        max_nudging_coefficient=interpolation_config.max_nudging_coefficient,
    )

    advection_config = advection.AdvectionConfig.from_fortran_dict(atmo_dict)

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig.from_fortran_dict(atmo_dict)

    return definitions.ExperimentConfig(
        driver=driver_cfg,
        vertical_grid=vertical_grid_config,
        nonhydrostatic=nonhydro_config,
        diffusion=diffusion_config,
        advection=advection_config,
        metrics=metrics_config,
        interpolation=interpolation_config,
        graupel=graupel_config,
        file_path=config_path,
    )
