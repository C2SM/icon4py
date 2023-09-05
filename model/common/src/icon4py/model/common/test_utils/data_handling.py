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

import wget


def download_and_extract(uri: str, local_path: Path, data_file: str):
    local_path.mkdir(parents=True, exist_ok=True)
    if not any(local_path.iterdir()):
        print(f"directory {local_path} is empty: downloading data from {uri} and extracting")
        wget.download(uri, out=data_file)
        # extract downloaded file
        if not tarfile.is_tarfile(data_file):
            raise NotImplementedError(f"{data_file} needs to be a valid tar file")
        with tarfile.open(data_file, mode="r:*") as tf:
            tf.extractall(path=local_path)
        Path(data_file).unlink(missing_ok=True)
