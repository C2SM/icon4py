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


import pytest

from atm_dyn_iconam.tests.test_utils.data_handling import download_and_extract
from atm_dyn_iconam.tests.test_utils.fixtures import base_path, data_uris
from icon4py.decomposition.parallel_setup import get_processor_properties


props = get_processor_properties(with_mpi=True)
path = base_path.joinpath("mpitask{props.comm_size}")
data_path = path.joinpath("mch_ch_r04b09_dsl/ser_data")


@pytest.fixture(scope="session")
def download_data():
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        uri = data_uris[props.comm_size]

        data_file = data_path.joinpath(
            f"mch_ch_r04b09_dsl_mpitask{props.comm_size}.tar.gz"
        ).name
        if props.rank == 0:
            download_and_extract(uri, path, data_file)
        props.comm.barrier()
    except KeyError:
        raise AssertionError(
            f"no data for communicator of size {props.comm_size} exists, use 1, 2 or 4"
        )
