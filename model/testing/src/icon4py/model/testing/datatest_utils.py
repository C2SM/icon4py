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
import ast
import re


import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, serialbox


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_experiment_name_with_version(experiment: definitions.Experiment) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment.name}_v{experiment.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment: definitions.Experiment, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment)}"


def get_experiment_archive_filename(experiment: definitions.Experiment, comm_size: int) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment, comm_size)}.tar.gz"


def get_serialized_data_url(root_url: str, filepath: str) -> str:
    """Build a download URL for serialized data file from root URL."""
    return f"{root_url}/download?path=%2F&files={urllib.parse.quote(filepath)}"


def get_datapath_for_experiment(
    experiment: definitions.Experiment,
    processor_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_dir = get_ranked_experiment_name_with_version(
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


def read_namelist(path: pathlib.Path) -> dict:
    """ICON NAMELIST_ICON_output_atm reader.
    Returns a dictionary of dictionaries, where the keys are the namelist names
    and the values are dictionaries of key-value pairs from the namelist.

    Use as:
    namelists = read_namelist("/path/to/NAMELIST_ICON_output_atm")
    print(namelists["NWP_TUNING_NML"]["TUNE_ZCEFF_MIN"])
    """
    with path.open() as f:
        txt = f.read()
    namelist_set = re.findall(r"&(\w+)(.*?)\/", txt, re.S)
    full_namelist = {}
    for namelist_name, namelist_content in namelist_set:
        full_namelist[namelist_name] = _parse_namelist_content(namelist_content)
    return full_namelist


def _parse_namelist_content(namelist_content):
    """
    Parse the contents of a single namelist group to a Python dictionary.
    """
    result = {}
    current_variable = None

    # Remove comments
    namelist_content = re.sub(r"!.*", "", namelist_content)

    # Split into lines
    lines = namelist_content.splitlines()

    for line in lines:
        line = line.strip()
        # skip any non-meaningful empty line
        if not line:
            continue

        # Remove trailing comma
        if line.endswith(","):
            line = line[:-1]

        if "=" in line:
            # New variable-value pair
            variable, value = line.split("=", 1)
            current_variable = variable.strip()
            result[current_variable] = _parse_value(value)
        # TODO(Chia Rui): check if continuation lines is irrelevant in our tests
        # else:
        #     # Continuation line (array or multi-line string)
        #     if current_variable is not None:
        #         val = _parse_value(line)
        #         # convert to a list if we have multiple values for the same variable
        #         if not isinstance(result[current_variable], list):
        #             result[current_variable] = [result[current_variable]]
        #         result[current_variable].append(val)

    return result


def _parse_value(value):
    """
    Convert Fortran-style values to Python types.
    """
    value = value.strip()

    # Fortran logical
    if value in ("T", ".T."):
        return True
    if value in ("F", ".F."):
        return False

    # Quoted string
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].rstrip()

    # Try numeric conversion
    try:
        return ast.literal_eval(value)
    except Exception:
        return value
