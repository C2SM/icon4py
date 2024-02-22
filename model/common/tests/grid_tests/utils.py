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
from pathlib import Path

from icon4py.model.common.test_utils.data_handling import download_and_extract
from icon4py.model.common.test_utils.datatest_utils import TEST_DATA_ROOT


MCH_GRID_FILE = "mch_ch_r04b09_dsl"
R02B04_GLOBAL = "r02b04_global"

grids_path = TEST_DATA_ROOT.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath(MCH_GRID_FILE)
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name

r02b04_global_grid_path = grids_path.joinpath(R02B04_GLOBAL)
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.tar.gz").name


MC_CH_R04B09_DSL_GRID_URI = "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
R02B04_GLOBAL_GRID_URI = "https://polybox.ethz.ch/index.php/s/AKAO6ImQdIatnkB/download"


def resolve_file_from_gridfile_name(name: str) -> Path:
    if name == MCH_GRID_FILE:
        gridfile = r04b09_dsl_grid_path.joinpath("grid.nc")
        if not gridfile.exists():
            download_and_extract(
                MC_CH_R04B09_DSL_GRID_URI,
                r04b09_dsl_grid_path,
                r04b09_dsl_grid_path,
                r04b09_dsl_data_file,
            )
        return gridfile
    elif name == R02B04_GLOBAL:
        gridfile = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.nc")
        if not gridfile.exists():
            download_and_extract(
                R02B04_GLOBAL_GRID_URI,
                r02b04_global_grid_path,
                r02b04_global_grid_path,
                r02b04_global_data_file,
            )
        return gridfile
    else:
        raise ValueError(f"invalid name: use one of {R02B04_GLOBAL, MCH_GRID_FILE}")
