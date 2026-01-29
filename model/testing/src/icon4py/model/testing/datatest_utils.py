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


def guess_grid_shape(experiment: definitions.Experiment) -> icon.GridShape:
    """Guess the grid type, root, and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
        RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron grid construction.

        Args: experiment: str: The experiment name.
        Returns: tuple[int, int]: The grid root and level.
    """
    if "torus" in experiment.name.lower():
        return icon.GridShape(geometry_type=base.GeometryType.TORUS)

    try:
        root, level = map(int, re.search(r"[Rr](\d+)[Bb](\d+)", experiment.name).groups())  # type:ignore[union-attr]
        return icon.GridShape(
            geometry_type=base.GeometryType.ICOSAHEDRON,
            subdivision=icon.GridSubdivision(root=root, level=level),
        )
    except AttributeError as err:
        raise ValueError(
            f"Could not parse grid_root and grid_level from experiment: {experiment.name} no 'rXbY'pattern."
        ) from err


def get_processor_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path: pathlib.Path, comm_size: int) -> pathlib.Path:
    return base_path.absolute().joinpath(f"mpitask{comm_size}")


def get_datapath_subdir_for_experiment(
    experiment: definitions.Experiment = definitions.Experiments.MCH_CH_R04B09,
) -> pathlib.Path:
    return pathlib.Path(f"{experiment.name}/ser_data")


def get_datapath_for_experiment(
    ranked_base_path: pathlib.Path,
    experiment: definitions.Experiment = definitions.Experiments.MCH_CH_R04B09,
) -> pathlib.Path:
    return ranked_base_path.joinpath(get_datapath_subdir_for_experiment(experiment))


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
