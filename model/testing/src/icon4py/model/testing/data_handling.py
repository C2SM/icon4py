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

from icon4py.model.testing import config, locking
from icon4py.model.testing.definitions import Experiment


def experiment_name_with_version(exp: Experiment) -> str:
    """Generate experiment name with version suffix.

    Args:
        experiment: Experiment object

    Returns:
        Name with version in format '{name}_v{version:02d}'
    """
    return f"{exp.name}_v{exp.version:02d}"


def experiment_archive_filename(exp: Experiment, comm_size: int | None = None) -> str:
    """Generate archive filename for an experiment.

    Args:
        experiment: Experiment object
        comm_size: Optional communicator size to prepend to filename

    Returns:
        Archive filename in format 'mpitaskX_{name}_v{version:02d}.tar.gz' if comm_size is provided,
        otherwise '{name}_v{version:02d}.tar.gz'
    """
    base_name = experiment_name_with_version(exp)
    if comm_size is not None:
        return f"mpitask{comm_size}_{base_name}.tar.gz"
    return f"{base_name}.tar.gz"


def download_and_extract(uri: str, dst: pathlib.Path, data_file: str = "downloaded.tar.gz") -> None:
    """
    Download data archive from remote server.

    Downloads a tar file at `uri` and extracts it.
    Args:
        uri: download url for archived data
        destination: the archive is extracted at this path
        data_file: filename of the downloaded archive, the archive is removed after download
    """
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import wget  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError(f"To download data file from {uri}, please install `wget`") from err

    wget.download(uri, out=data_file)
    if not tarfile.is_tarfile(data_file):
        raise OSError(f"{data_file} needs to be a valid tar file")
    with tarfile.open(data_file, mode="r:*") as tf:
        tf.extractall(path=dst)
    pathlib.Path(data_file).unlink(missing_ok=True)


def download_test_data(dst: pathlib.Path, uri: str) -> None:
    if config.ENABLE_TESTDATA_DOWNLOAD:
        dst.mkdir(parents=True, exist_ok=True)
        # Explicitly specify the lockfile name to make sure that os.listdir sees
        # it if it's created in dst.
        lockfile = "filelock.lock"
        with locking.lock(dst, lockfile=lockfile):
            files = os.listdir(dst)
            if len(files) == 0 or (len(files) == 1 and files[0] == lockfile):
                download_and_extract(uri, dst)
    else:
        # If test data download is disabled, we check if the directory exists
        # and isn't empty without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not dst.exists():
            raise RuntimeError(f"Test data {dst} does not exist, and downloading is disabled.")
        elif not any(os.scandir(dst)):
            raise RuntimeError(f"Test data {dst} exists but is empty, and downloading is disabled.")
