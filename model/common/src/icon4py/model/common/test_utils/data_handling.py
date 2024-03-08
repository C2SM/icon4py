# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import tarfile
from pathlib import Path


def download_and_extract(
    uri: str, base_path: Path, destination_path: Path, data_file: str = "downloaded.tar.gz"
):
    """
    "Download data archive from remote server.

    Checks whether a given directory `destination_path` is empty and if so downloads a the tar
    file at `uri` and extracts it.
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
            Path(data_file).unlink(missing_ok=True)
        except ImportError as err:
            raise FileNotFoundError(
                f" To download data file from {uri}, please install `wget`"
            ) from err
