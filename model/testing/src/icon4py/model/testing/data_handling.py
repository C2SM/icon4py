# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import tarfile
from pathlib import Path


def download_and_extract(uri: str, dst: Path, data_file: str = "downloaded.tar.gz") -> None:
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
        raise IOError(f"{data_file} needs to be a valid tar file")
    with tarfile.open(data_file, mode="r:*") as tf:
        tf.extractall(path=dst)
    Path(data_file).unlink(missing_ok=True)
