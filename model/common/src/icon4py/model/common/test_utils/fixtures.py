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

from ..decomposition.parallel_setup import get_processor_properties
from .data_handling import download_and_extract
from .serialbox_utils import IconSerialDataProvider
from .simple_mesh import SimpleMesh


test_utils = Path(__file__).parent
model = test_utils.parent.parent
common = model.parent.parent.parent.parent
base_path = common.parent.joinpath("testdata")

data_uris = {
    1: "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download",
    2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download",
    4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download",
}

ser_data_basepath = base_path.joinpath("ser_icondata")


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    with_mpi = request.param
    return get_processor_properties(with_mpi=with_mpi)


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
    return ser_data_basepath.absolute().joinpath(f"mpitask{processor_props.comm_size}")


@pytest.fixture(scope="session")
def datapath(ranked_data_path):
    return ranked_data_path.joinpath("mch_ch_r04b09_dsl/ser_data")


@pytest.fixture(scope="session")
def download_ser_data(request, processor_props, ranked_data_path, pytestconfig):
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    if not pytestconfig.getoption("datatest"):
        pytest.skip("not running datatest marked tests")

    try:
        uri = data_uris[processor_props.comm_size]

        data_file = ranked_data_path.joinpath(
            f"mch_ch_r04b09_dsl_mpitask{processor_props.comm_size}.tar.gz"
        ).name
        if processor_props.rank == 0:
            download_and_extract(uri, ranked_data_path, data_file)
        if processor_props.comm:
            processor_props.comm.barrier()
    except KeyError:
        raise AssertionError(
            f"no data for communicator of size {processor_props.comm_size} exists, use 1, 2 or 4"
        )


@pytest.fixture(scope="session")
def data_provider(
    download_ser_data, datapath, processor_props
) -> IconSerialDataProvider:
    return IconSerialDataProvider(
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=processor_props.rank,
        do_print=True,
    )


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


@pytest.fixture
def interpolation_savepoint(data_provider):  # F811
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):  # F811
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()

@pytest.fixture
def metrics_nonhydro_savepoint(data_provider):  # noqa F811
    """Load data from ICON metric state nonhydro savepoint."""
    return data_provider.from_metrics_nonhydro_savepoint()



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


@pytest.fixture
def savepoint_velocity_init(
    data_provider, step_date_init, istep, vn_only, jstep  # noqa F811
):
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep, vn_only=vn_only, date=step_date_init, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_init(data_provider, step_date_init, istep, jstep):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(
        istep=istep, date=step_date_init, jstep=jstep
    )


@pytest.fixture
def savepoint_velocity_exit(
    data_provider, step_date_exit, istep, vn_only, jstep  # noqa F811
):
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep, vn_only=vn_only, date=step_date_exit, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep, jstep):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(
        istep=istep, date=step_date_exit, jstep=jstep
    )


@pytest.fixture
def savepoint_nonhydro_step_exit(data_provider, step_date_exit, jstep):  # noqa F811
    """
    Load data from ICON savepoint at final exit (after predictor and corrector, and 3 final stencils) of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_step_exit(
        date=step_date_exit, jstep=jstep
    )



@pytest.fixture
def istep():
    return 1


@pytest.fixture
def jstep():
    return 0


@pytest.fixture
def ntnd(savepoint_velocity_init):
    return savepoint_velocity_init.get_metadata("ntnd").get("ntnd")


@pytest.fixture
def vn_only():
    return False
