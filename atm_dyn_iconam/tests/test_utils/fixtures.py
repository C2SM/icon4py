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


base_path = Path(__file__).parent.parent.parent.parent.joinpath("testdata")

data_uris = {
    1: "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download",
    2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download",
    4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download",
}
mch_ch_r04b09_dsl_grid_uri = (
    "https://polybox.ethz.ch/index.php/s/hD232znfEPBh4Oh/download"
)
r02b04_global_grid_uri = "https://polybox.ethz.ch/index.php/s/0EM8O8U53GKGsst/download"


data_path = base_path.joinpath("ser_icondata")
data_file = data_path.joinpath("mch_ch_r04b09_dsl.tar.gz").name

grids_path = base_path.joinpath("grids")
r04b09_dsl_grid_path = grids_path.joinpath("mch_ch_r04b09_dsl")
r04b09_dsl_data_file = r04b09_dsl_grid_path.joinpath(
    "mch_ch_r04b09_dsl_grids_v1.tar.gz"
).name
r02b04_global_grid_path = grids_path.joinpath("r02b04_global")
r02b04_global_data_file = r02b04_global_grid_path.joinpath(
    "icon_grid_0013_R02B04_G.tar.gz"
).name


@pytest.fixture
def datapath(setup_icon_data):
    local_path = "mpitask1/mch_ch_r04b09_dsl/ser_data"
    return data_path.joinpath(local_path)


@pytest.fixture(scope="session")
def setup_icon_data():
    """
    Get the binary ICON data for single node run from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(data_uris[1], data_path, data_file)


@pytest.fixture
def data_provider(setup_icon_data, datapath) -> IconSerialDataProvider:
    path = str(datapath)
    return IconSerialDataProvider("icon_pydycore", path, True)


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid()


@pytest.fixture
def damping_height():
    return 12500


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
