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

import pooch

from icon4py.model.testing import config, locking


def download_and_extract(
    uri: str,
    dst: pathlib.Path,
    known_hash: str | None,
) -> None:
    """
    Download and extract a tar file with locking.

    Args:
        uri: download url for archived data
        dst: the archive is extracted at this path
        known_hash: expected hash of the archive for integrity verification,
            or None to skip verification

    Uses pooch for downloading and archive extraction.
    """
    dst.mkdir(parents=True, exist_ok=True)

    completion_marker = dst / ".extraction_complete"
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
        pooch.retrieve(
            url=uri,
            known_hash=known_hash,
            path=str(dst),
            fname="archive.tar.gz",
            processor=pooch.Untar(extract_dir="."),
        )
        completion_marker.touch()


def download_test_data(dst: pathlib.Path, uri: str, known_hash: str | None) -> None:
    if config.ENABLE_TESTDATA_DOWNLOAD:
        download_and_extract(uri, dst, known_hash=known_hash)
    else:
        # If test data download is disabled, we check if the directory exists
        # and isn't empty without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not dst.exists():
            raise RuntimeError(f"Test data {dst} does not exist, and downloading is disabled.")
        elif not any(os.scandir(dst)):
            raise RuntimeError(f"Test data {dst} exists but is empty, and downloading is disabled.")
