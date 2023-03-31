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
import sys
from pathlib import Path

from gt4py.next.common import Dimension

from icon4py.icon4pygen.icochainsize import IcoChainSize


PYTHON_PATH = sys.executable


def check_dir_exists(dirpath: Path) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)


def write_string(string: str, outdir: Path, fname: str) -> None:
    path = outdir / fname
    with open(path, "w") as f:
        f.write(string)


def calc_num_neighbors(dim_list: list[Dimension], includes_center: bool) -> int:
    return IcoChainSize.get(dim_list) + int(includes_center)


def format_fortran_code(source: str) -> str:
    """Format fortran code using fprettify."""
    bin_path = Path(PYTHON_PATH).parent
    if "spack" not in bin_path:
        fprettify_path = bin_path / "fprettify"
    else:
        fprettify_path = "fprettify"
    args = [str(fprettify_path)]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return p1.communicate(source.encode("UTF-8"))[0].decode("UTF-8").rstrip()
