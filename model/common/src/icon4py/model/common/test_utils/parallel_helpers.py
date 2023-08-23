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

from icon4py.model.common.decomposition.parallel_setup import ProcessProperties
from icon4py.model.common.test_utils.data_handling import download_and_extract
from icon4py.model.common.test_utils.fixtures import data_path, data_uris




@pytest.fixture(scope="session")
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def download_data(request, processor_props, ranked_data_path):
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        uri = data_uris[processor_props.comm_size]

        data_file = data_path.joinpath(
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





def check_comm_size(props:ProcessProperties, sizes=[1,2,4]):
    if props.comm_size not in sizes:
        pytest.xfail(f"wrong comm size: {props.comm_size}: test only works for sizes: {sizes}")

