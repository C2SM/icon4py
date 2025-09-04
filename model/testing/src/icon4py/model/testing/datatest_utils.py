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
import uuid

from gt4py.next import backend as gtx_backend

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, icon
from icon4py.model.testing import definitions, serialbox


GLOBAL_EXPERIMENT = definitions.Experiments.EXCLAIM_APE.name
REGIONAL_EXPERIMENT = definitions.Grids.MCH_CH_R04B09_DSL.name
# we were using `REGIONAL_EXPERIMENT` for both grid and experiment, therefore we make sure their names match
assert definitions.Grids.MCH_CH_R04B09_DSL.name == definitions.Experiments.MCH_CH_R04B09.name
R02B04_GLOBAL = definitions.Grids.R02B04_GLOBAL.name
R02B07_GLOBAL = definitions.Grids.R02B07_GLOBAL.name
ICON_CH2_SMALL = definitions.Grids.MCH_OPR_R04B07_DOMAIN01.name
REGIONAL_BENCHMARK = definitions.Grids.MCH_OPR_R19B08_DOMAIN01.name
JABW_EXPERIMENT = definitions.Experiments.JW.name
GAUSS3D_EXPERIMENT = definitions.Experiments.GAUSS3D.name
WEISMAN_KLEMP_EXPERIMENT = definitions.Experiments.WEISMAN_KLEMP_TORUS.name


# maps experiment or grid to grid URI
GRID_URIS = {
    REGIONAL_EXPERIMENT: definitions.Grids.MCH_CH_R04B09_DSL.uri,
    R02B04_GLOBAL: definitions.Grids.R02B04_GLOBAL.uri,
    R02B07_GLOBAL: definitions.Grids.R02B07_GLOBAL.uri,
    ICON_CH2_SMALL: definitions.Grids.MCH_OPR_R04B07_DOMAIN01.uri,
    REGIONAL_BENCHMARK: definitions.Grids.MCH_OPR_R19B08_DOMAIN01.uri,
    WEISMAN_KLEMP_EXPERIMENT: definitions.Grids.TORUS_50000x5000.uri,
}

# TODO(havogt): is this still needed?
GRID_IDS = {
    GLOBAL_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    REGIONAL_EXPERIMENT: uuid.UUID("f2e06839-694a-cca1-a3d5-028e0ff326e0"),
    JABW_EXPERIMENT: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    GAUSS3D_EXPERIMENT: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
    WEISMAN_KLEMP_EXPERIMENT: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
}


def guess_grid_shape(experiment: str) -> icon.GridShape:
    """Guess the grid type, root, and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
        RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron grid construction.

        Args: experiment: str: The experiment name.
        Returns: tuple[int, int]: The grid root and level.
    """
    if "torus" in experiment.lower():
        return icon.GridShape(geometry_type=base.GeometryType.TORUS)

    try:
        root, level = map(int, re.search(r"[Rr](\d+)[Bb](\d+)", experiment).groups())  # type:ignore[union-attr]
        return icon.GridShape(
            geometry_type=base.GeometryType.ICOSAHEDRON,
            subdivision=icon.GridSubdivision(root=root, level=level),
        )
    except AttributeError as err:
        raise ValueError(
            f"Could not parse grid_root and grid_level from experiment: {experiment} no 'rXbY'pattern."
        ) from err


def get_grid_id_for_experiment(experiment: str) -> uuid.UUID:
    """Get the unique id of the grid used in the experiment.

    These ids are encoded in the original grid file that was used to run the simulation, but not serialized when generating the test data. So we duplicate the information here.

    TODO(halungge): this becomes obsolete once we get the connectivities from the grid files.
    """
    try:
        return GRID_IDS[experiment]
    except KeyError as err:
        raise ValueError(f"Experiment '{experiment}' has no grid id ") from err


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
    backend: gtx_backend.Backend | None,
) -> serialbox.IconSerialDataProvider:
    return serialbox.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=rank,
        do_print=True,
    )


# TODO(havogt): this function should disappear after the refactoring from raw string to Experiment is completed
def experiment_from_name(experiment_name: str) -> definitions.Experiment:
    for item in vars(definitions.Experiments).values():
        if isinstance(item, definitions.Experiment) and item.name == experiment_name:
            return item
    raise ValueError(f"No such experiment: {experiment_name}")
