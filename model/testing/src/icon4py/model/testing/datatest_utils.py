# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
import re

import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, icon
from icon4py.model.testing import definitions, serialbox


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path: pathlib.Path, comm_size: int) -> pathlib.Path:
    return base_path.absolute().joinpath(f"mpitask{comm_size}")


def get_datapath_for_experiment(
    ranked_base_path: pathlib.Path,
    experiment: definitions.Experiment = definitions.Experiments.MCH_CH_R04B09,
) -> pathlib.Path:
    return ranked_base_path.joinpath(f"{experiment.name}/ser_data")


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
