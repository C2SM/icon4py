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

import subprocess
from pathlib import Path


def run_subprocess(*args, **kwargs) -> None:
    """Run a command using the given positional and keyword arguments."""
    result = subprocess.run(
        *args, **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode != 0:
        error = result.stdout.decode()
        raise RuntimeError(error)


def check_dir_exists(dirpath: Path) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)


def write_string(string: str, outdir: Path, fname: str) -> None:
    path = outdir / fname
    with open(path, "w") as f:
        f.write(string)
