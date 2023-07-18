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

import pytest

from .data_handling import download_and_extract
from .serialbox_utils import IconSerialDataProvider


data_uri = "https://polybox.ethz.ch/index.php/s/LcAbscZqnsx4WCf/download"
data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")
data_file = data_path.joinpath("mch_ch_r04b09_dsl_v2.tar.gz").name
mch_ch_r04b09_dsl_grid_uri = (
    "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
)

r02b04_global_grid_uri = "https://polybox.ethz.ch/index.php/s/0EM8O8U53GKGsst/download"
grids_path = Path(__file__).parent.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath("mch_ch_r04b09_dsl")
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath(
    "mch_ch_r04b09_dsl_grids_v1.tar.gz"
).name
r02b04_global_grid_path = grids_path.joinpath("icon_r02b04_global")
r02b04_global_data_file = r02b04_global_grid_path.joinpath(
    "icon_grid_0013_R02B04_G.tar.gz"
).name


@pytest.fixture(scope="session")
def setup_icon_data():
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(data_uri, data_path, data_file)


@pytest.fixture
def data_provider(setup_icon_data) -> IconSerialDataProvider:
    return IconSerialDataProvider("icon_pydycore", str(extracted_path), True)


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture(scope="session")
def get_grid_files():
    """
    Get the grid files used for testing.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(
        mch_ch_r04b09_dsl_grid_uri, r04b09_dsl_grid_path, r04b09_dsl_data_file
    )
    download_and_extract(
        r02b04_global_grid_uri, r02b04_global_grid_path, r02b04_global_data_file
    )


@pytest.fixture()
def r04b09_dsl_gridfile(get_grid_files):
    return r04b09_dsl_grid_path.joinpath("grid.nc")
