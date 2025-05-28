# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import re
import uuid
from typing import Final, Optional

from gt4py.next import backend as gtx_backend

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing.definitions import (
    Experiment.GAUSS3D,
    Experiment.GLOBAL,
    Experiment.JABW,
    Experiment.REGIONAL,
    Experiment.WEISMAN_KLEMP,
)


_TEST_UTILS_PATH: Final = pathlib.Path(__file__) / ".."
_MODEL_PATH: Final = _TEST_UTILS_PATH / ".."
_COMMON_PATH: Final = _MODEL_PATH / ".." / ".." / ".." / ".."

DEFAULT_TEST_DATA_FOLDER: Final = "testdata"

TEST_DATA_ROOT: Final[pathlib.Path] = pathlib.Path(
    os.getenv("TEST_DATA_PATH", _COMMON_PATH / ".." / (DEFAULT_TEST_DATA_FOLDER))
)
SERIALIZED_DATA_PATH: Final[pathlib.Path] = TEST_DATA_ROOT / "ser_icondata"
GRIDS_PATH: Final[pathlib.Path] = TEST_DATA_ROOT / "grids"

GRID_IDS = {
    Experiment.GLOBAL: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    Experiment.REGIONAL: uuid.UUID("f2e06839-694a-cca1-a3d5-028e0ff326e0"),
    Experiment.JABW: uuid.UUID("af122aca-1dd2-11b2-a7f8-c7bf6bc21eba"),
    Experiment.GAUSS3D: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
    Experiment.WEISMAN_KLEMP: uuid.UUID("80ae276e-ec54-11ee-bf58-e36354187f08"),
}


def get_global_grid_params(experiment: str) -> tuple[int, int]:
    """
    Get the grid root and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
    RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron
    grid construction.

    Args:
        experiment: The experiment name

    Returns:
        The grid root and level

    """
    if "torus" in experiment:
        # these magic values seem to mark a torus: they are set in all torus grid files.
        return 0, 2

    try:
        root, level = [int(i) for i in re.search("[Rr](\d+)[Bb](\d+)", experiment).groups()]
        return root, level
    except AttributeError as err:
        raise ValueError(
            f"Could not parse grid_root and grid_level from experiment: {experiment} no 'rXbY'pattern."
        ) from err


def get_grid_id_for_experiment(experiment) -> uuid.UUID:
    """Get the unique id of the grid used in the experiment.

    These ids are encoded in the original grid file that was used to run the simulation, but not serialized when generating the test data. So we duplicate the information here.

    TODO (@halungge): this becomes obsolete once we get the connectivities from the grid files.
    """
    try:
        return GRID_IDS[experiment]
    except KeyError as err:
        raise ValueError(f"Experiment '{experiment}' has no grid id ") from err


def get_processor_properties_for_run(run_instance):
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path, processor_properties):
    return base_path.absolute().joinpath(f"mpitask{processor_properties.comm_size}")


def get_datapath_for_experiment(ranked_base_path, experiment=Experiment.REGIONAL):
    return ranked_base_path.joinpath(f"{experiment}/ser_data")


def create_icon_serial_data_provider(
    datapath, processor_props, backend: Optional[gtx_backend.Backend]
):
    # note: this needs to be here, otherwise spack doesn't find serialbox
    from icon4py.model.testing.serialbox import IconSerialDataProvider

    return IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )
