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
import tempfile

from icon4py.model.testing import config, locking


def download_and_extract(uri: str, dst: pathlib.Path) -> None:
    """
    Download data archive from remote server.

    Downloads a tar file at `uri` and extracts it.
    Args:
        uri: download url for archived data
        dst: the archive is extracted at this path

    Uses a temporary file for the download, which is automatically cleaned up
    on completion or interruption, enabling safe parallel execution.
    """
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import wget  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError(f"To download data file from {uri}, please install `wget`") from err

    with tempfile.NamedTemporaryFile(delete=True, suffix=".tar.gz") as temp_file:
        wget.download(uri, out=temp_file.name)
        if not tarfile.is_tarfile(temp_file.name):
            raise OSError(f"{temp_file.name} needs to be a valid tar file")
        with tarfile.open(temp_file.name, mode="r:*") as tf:
            tf.extractall(path=dst)


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
