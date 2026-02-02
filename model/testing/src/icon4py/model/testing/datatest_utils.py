# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
import urllib.parse

import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, serialbox
from icon4py.model.testing.definitions import Experiment


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path: pathlib.Path, comm_size: int) -> pathlib.Path:
    return base_path.absolute().joinpath(f"mpitask{comm_size}")


def experiment_name_with_version(experiment: Experiment) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment.name}_v{experiment.version:02d}"


def ranked_experiment_name_with_version(experiment: Experiment, comm_size: int) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{experiment_name_with_version(experiment)}"


def experiment_archive_filename(experiment: Experiment, comm_size: int) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{ranked_experiment_name_with_version(experiment, comm_size)}.tar.gz"


def build_serialized_data_url(root_url: str, filepath: str) -> str:
    """Build a download URL for serialized data file from root URL.

    Args:
        root_url: Root polybox URL (without /download?path=...)
        filepath: Path of the file to download (e.g., ser_icondata/mpitask1_expname_v00.tar.gz)

    Returns:
        Complete download URL with filename parameter
    """

    return f"{root_url}/download?path=%2F&files={urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment.

    With the flattened structure, data for an experiment is stored as:
        base_path/mpitaskX_experiment_name_version/ser_data

    Args:
        experiment: Experiment to get data path for

    Returns:
        Path to the ser_data directory for the experiment
    """

    # Construct the ranked directory name: mpitaskX_expname_vYY
    experiment_dir = ranked_experiment_name_with_version(
        experiment,
        processor_props.comm_size,
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
