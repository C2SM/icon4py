# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.data_handling import download_and_extract
from icon4py.model.common.test_utils.datatest_utils import (
    GRIDS_PATH,
    MC_CH_R04B09_DSL_GRID_URI,
    R02B04_GLOBAL,
    R02B04_GLOBAL_GRID_URI,
    REGIONAL_EXPERIMENT,
)


r04b09_dsl_grid_path = GRIDS_PATH.joinpath(REGIONAL_EXPERIMENT)
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath("mch_ch_r04b09_dsl_grids_v1.tar.gz").name

r02b04_global_grid_path = GRIDS_PATH.joinpath(R02B04_GLOBAL)
r02b04_global_data_file = r02b04_global_grid_path.joinpath("icon_grid_0013_R02B04_R.tar.gz").name


def resolve_file_from_gridfile_name(name: str) -> Path:
    if name == REGIONAL_EXPERIMENT:
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
        raise ValueError(f"invalid name: use one of {R02B04_GLOBAL, REGIONAL_EXPERIMENT}")


def horizontal_dim():
    for dim in (dims.VertexDim, dims.EdgeDim, dims.CellDim):
        yield dim
