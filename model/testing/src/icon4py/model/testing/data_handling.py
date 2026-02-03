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
import shutil
import tarfile
import tempfile

from icon4py.model.testing import config, locking


def download_and_extract(
    uri: str,
    dst: pathlib.Path,
) -> None:
    """
    Download and extract a tar file with locking and retry logic.

    Args:
        uri: download url for archived data
        dst: the archive is extracted at this path

    Downloads to a temporary directory in the destination directory
    (not /tmp to avoid space constraints).
    """
    dst.mkdir(parents=True, exist_ok=True)

    completion_marker = dst / ".download_complete"
    lockfile = "filelock.lock"

    with locking.lock(dst, lockfile=lockfile):
        if completion_marker.exists():
            return
        # Clean up any partial data from previous failed attempts
        # (except for the lockfile)
        for item in dst.iterdir():
            if item.name != lockfile:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        _perform_download(uri, dst)
        completion_marker.touch()


def _perform_download(uri: str, dst: pathlib.Path) -> None:
    try:
        import wget  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError(f"To download data file from {uri}, please install `wget`") from err

    with tempfile.TemporaryDirectory(dir=dst) as temp_dir:
        temp_path = pathlib.Path(temp_dir) / "download.tar.gz"
        wget.download(uri, out=str(temp_path))
        if not tarfile.is_tarfile(temp_path):
            raise OSError(f"{temp_path} needs to be a valid tar file")
        with tarfile.open(temp_path, mode="r:*") as tf:
            tf.extractall(path=dst)


def download_test_data(dst: pathlib.Path, uri: str) -> None:
    if config.ENABLE_TESTDATA_DOWNLOAD:
        download_and_extract(uri, dst)
    else:
        # If test data download is disabled, we check if the directory exists
        # and isn't empty without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not dst.exists():
            raise RuntimeError(f"Test data {dst} does not exist, and downloading is disabled.")
        elif not any(os.scandir(dst)):
            raise RuntimeError(f"Test data {dst} exists but is empty, and downloading is disabled.")
