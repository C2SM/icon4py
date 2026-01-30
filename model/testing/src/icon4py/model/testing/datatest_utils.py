# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib

import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import definitions, serialbox


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path: pathlib.Path, comm_size: int) -> pathlib.Path:
    return base_path.absolute().joinpath(f"mpitask{comm_size}")


#TODO (jcanton): pass comm_size directly instead of ranked_base_path
def get_datapath_for_experiment(
    ranked_base_path: pathlib.Path,
    experiment: definitions.Experiment = definitions.Experiments.MCH_CH_R04B09,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment.
    
    With the flattened structure, data for an experiment is stored as:
        base_path/mpitaskX_experiment_name_version/ser_data
    
    Args:
        ranked_base_path: Path like ser_icondata/mpitaskX, used to extract rank info
        experiment: Experiment to get data path for
        
    Returns:
        Path to the ser_data directory for the experiment
    """
    from icon4py.model.testing.data_handling import experiment_name_with_version
    
    # Extract rank from the path name (e.g., mpitask2 -> 2)
    path_str = ranked_base_path.name
    rank = path_str.split("mpitask")[1] if "mpitask" in path_str else ""
    
    # Construct the directory name: mpitaskX_expname_vYY
    exp_dir = f"mpitask{rank}_{experiment_name_with_version(experiment)}"
    
    return ranked_base_path.parent.joinpath(exp_dir, definitions.SERIALIZED_DATA_SUBDIR)


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
