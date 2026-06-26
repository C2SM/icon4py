# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
import logging
import pathlib
from typing import Any

from icon4py.model.experiment_config.config import (
    AbsoluteTime,
    DriverConfig,
    EndOfSimulation,
    ExperimentConfig,
    NumTimeSteps,
    ProfilingStats,
    RelativeTime,
)
from icon4py.model.experiment_config.reader import read_experiment_config


# These types and the reader now live in the shared ``experiment_config`` package; they are
# re-exported here so that existing ``driver_config.<name>`` references keep working.
__all__ = [
    "AbsoluteTime",
    "DriverConfig",
    "EndOfSimulation",
    "ExperimentConfig",
    "NumTimeSteps",
    "ProfilingStats",
    "RelativeTime",
    "prepare_output_directory",
    "read_config",
]

log = logging.getLogger(__name__)


def read_config(
    config_file_path: pathlib.Path,
    enable_profiling: bool = False,
) -> ExperimentConfig:
    return read_experiment_config(config_file_path, enable_profiling=enable_profiling)


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
