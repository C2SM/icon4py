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
import tarfile
from typing import Final, Optional

import wget

from icon4py.model.testing import cases, serialbox as testing_ser


ICON4PY_MODEL_TESTING_PKG_SRC_PATH: Final = pathlib.Path(__file__).parent.resolve()
REPO_ROOT_PATH: Final = (
    ICON4PY_MODEL_TESTING_PKG_SRC_PATH / ".." / ".." / ".." / ".." / ".." / ".."
).resolve()

TEST_DATA_PATH: Final[pathlib.Path] = pathlib.Path(
    os.getenv("ICON4PY_TEST_DATA_PATH", REPO_ROOT_PATH / "testdata")
).resolve()
GRIDS_PATH: Final[pathlib.Path] = (TEST_DATA_PATH / "grids").resolve()
SERIALIZED_DATA_PATH: Final[pathlib.Path] = (TEST_DATA_PATH / "icon_serialbox").resolve()


def download_and_extract(
    uri: str,
    target_path: pathlib.Path,
    *,
    temp_filename: str = "downloaded.tar.gz",
) -> None:
    """
    Download data archive from remote server.

    Check whether a given data content already exists and otherwise
    create it from a tar archive at `uri` and extract it.

    Args:
        uri: URL for data archive (`tar` format only).
        target_path: Expected final location of the extracted data.
        temp_filename: temporaty name of the downloaded archive (removed after extraction).
    """
    
    if not target_path.exists() or (target_path.is_dir() and not any(target_path.iterdir())):
        print(f"Downloading and extracting '{target_path}' data from '{uri}'")
        wget.download(uri, out=temp_filename)
        if not tarfile.is_tarfile(temp_filename):
            raise RuntimeError(f"{temp_filename} needs to be a valid tar file")

        base_path = target_path.parent    
        with tarfile.open(temp_filename, mode="r:*") as tf:
            tf.extractall(path=base_path)
        if not target_path.exists():
            raise RuntimeError(f"{target_path} does not exist after extraction!!")
        pathlib.Path(temp_filename).unlink(missing_ok=True)
        print("")



def get_grid(grid: cases.Grid) -> pathlib.Path:
    assert grid.download_file_path
    assert grid.uri

    grid_file_path = grid.download_file_path
    if not grid_file_path.exists():
        download_and_extract(grid.uri, grid_file_path)

    return grid_file_path


def get_processor_properties_for_run(run_instance):
    return decomposition.get_processor_properties(run_instance)


def get_ranked_data_path(base_path, processor_properties) -> pathlib.Path:
    return base_path.absolute().joinpath(f"mpitask{processor_properties.comm_size}")


def get_datapath_for_experiment(ranked_base_path, experiment) -> pathlib.Path:
    return ranked_base_path.joinpath(f"{experiment}/ser_data")


def create_icon_serial_data_provider(
    datapath, processor_props, backend: Optional["gtx_backend.Backend"]
):
    return testing_ser.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )
