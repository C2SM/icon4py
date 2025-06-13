# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
import re
import tarfile
import uuid
from typing import Final, Optional

from icon4py.model.testing import serialbox as icon4py_serialbox


ICON4PY_MODEL_TESTING_PKG_SRC_PATH: Final = pathlib.Path(__file__).parent.resolve()
REPO_ROOT_PATH: Final = (
    ICON4PY_MODEL_TESTING_PKG_SRC_PATH / ".." / ".." / ".." / ".." / ".." / ".."
).resolve()

ICON4PY_TEST_DATA_PATH: Final[pathlib.Path] = pathlib.Path(
    os.getenv("TEST_DATA_PATH") or (REPO_ROOT_PATH / "testdata")
).resolve()
GRIDS_PATH: Final[pathlib.Path] = (ICON4PY_TEST_DATA_PATH / "grids").resolve()
SERIALIZED_DATA_PATH: Final[pathlib.Path] = (ICON4PY_TEST_DATA_PATH / "icon_serialbox").resolve()


def download_and_extract(
    uri: str,
    base_path: pathlib.Path,
    destination_path: pathlib.Path,
    data_file: str = "downloaded.tar.gz",
) -> None:
    """
    "Download data archive from remote server.

    Check whether a given directory `destination_path` is empty and, if so,
    download a tar file at `uri` and extract it.

    Args:
        uri: download url for archived data
        base_path: the archive is extracted at this path it might be different from the final
            destination to account for directories in the archive
        destination_path: final expected location of the extracted data
        data_file: local final of the archive is removed after download
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    if not any(destination_path.iterdir()):
        try:
            import wget

            print(
                f"directory {destination_path} is empty: downloading data from {uri} and extracting"
            )
            wget.download(uri, out=data_file)
            # extract downloaded file
            if not tarfile.is_tarfile(data_file):
                raise NotImplementedError(f"{data_file} needs to be a valid tar file")
            with tarfile.open(data_file, mode="r:*") as tf:
                tf.extractall(path=base_path)
            pathlib.Path(data_file).unlink(missing_ok=True)
        except ImportError as err:
            raise FileNotFoundError(
                f" To download data file from {uri}, please install `wget`"
            ) from err


def get_global_grid_params(experiment: str) -> tuple[int, int]:
    """Get the grid root and level from the experiment name.

    Reads the level and root parameters from a string in the canonical ICON gridfile format
        RxyBab where 'xy' and 'ab' are numbers and denote the root and level of the icosahedron grid construction.

        Args: experiment: str: The experiment name.
        Returns: tuple[int, int]: The grid root and level.
    """
    if "torus" in experiment:
        # these magic values seem to mark a torus: they are set in all torus grid files.
        return 0, 2

    try:
        root, level = map(int, re.search("[Rr](\d+)[Bb](\d+)", experiment).groups())
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


def get_datapath_for_experiment(ranked_base_path, experiment):
    return ranked_base_path.joinpath(f"{experiment}/ser_data")


def create_icon_serial_data_provider(
    datapath, processor_props, backend: Optional["gtx_backend.Backend"]
):
    return icon4py_serialbox.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )
