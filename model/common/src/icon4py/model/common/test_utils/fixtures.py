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
from gt4py.next.program_processors.runners.roundtrip import executor

from .data_handling import download_and_extract
from .serialbox_utils import IconSerialDataProvider
from .simple_mesh import SimpleMesh
from ..decomposition.parallel_setup import get_processor_properties

test_utils = Path(__file__).parent
model = test_utils.parent.parent
common = model.parent.parent.parent.parent
base_path = common.parent.joinpath("testdata")

data_uris = {
    1: "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download",
    2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download",
    4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download",
}
mch_ch_r04b09_dsl_grid_uri = (
    "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download"
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


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    with_mpi = request.param
    return get_processor_properties(with_mpi=with_mpi)


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
     return data_path.absolute().joinpath(f"mpitask{processor_props.comm_size}")

@pytest.fixture(scope="session")
def datapath(setup_icon_data, ranked_data_path):
    return ranked_data_path.joinpath("mch_ch_r04b09_dsl/ser_data")


@pytest.fixture(scope="session")
def setup_icon_data():
    """
    Get the binary ICON data for single node run from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    download_and_extract(data_uris[1], data_path, data_file)

# TODO (magdalena) dependency on setup_icon_data or download_data
@pytest.fixture(scope="session")
def data_provider(setup_icon_data, datapath, processor_props) -> IconSerialDataProvider:
    return IconSerialDataProvider(fname_prefix="icon_pydycore", path=str(datapath), mpi_rank=processor_props.rank, do_print=True)


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
def decomposition_info(data_provider):
    return data_provider.from_savepoint_grid().construct_decomposition_info()

@pytest.fixture
def damping_height():
    return 12500



@pytest.fixture
def ndyn_substeps():
    """
    Return number of dynamical substeps.

    Serialized data uses a reduced number (2 instead of the default 5) in order to reduce the amount
    of data generated.
    """
    return 2


@pytest.fixture
def linit():
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init():
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def step_date_exit():
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


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


@pytest.fixture
def interpolation_savepoint(data_provider):  # noqa F811
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):  # noqa F811
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()



@pytest.fixture()
def r04b09_dsl_gridfile(get_grid_files):
    return r04b09_dsl_grid_path.joinpath("grid.nc")

BACKENDS = {"embedded": executor}
MESHES = {"simple_mesh": SimpleMesh()}


@pytest.fixture(
    ids=MESHES.keys(),
    params=MESHES.values(),
)
def mesh(request):
    return request.param


@pytest.fixture(ids=BACKENDS.keys(), params=BACKENDS.values())
def backend(request):
    return request.param
