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

from atm_dyn_iconam.tests.test_utils.data_handling import download_and_extract
from icon4py.decomposition.parallel_setup import get_processor_properties


props = get_processor_properties(with_mpi=True)
base_path = Path(__file__).parent.parent.parent.parent.joinpath(f"testdata/ser_icondata/mpitask{props.comm_size}/")
path = base_path.joinpath("mch_ch_r04b09_dsl/ser_data")
data_uris = {1: "https://polybox.ethz.ch/index.php/s/vcsCYmCFA9Qe26p/download"
, 2: "https://polybox.ethz.ch/index.php/s/NUQjmJcMEoQxFiK/download", 4: "https://polybox.ethz.ch/index.php/s/QC7xt7xLT5xeVN5/download"}

@pytest.fixture(scope="session")
def download_data():
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        uri = data_uris[props.comm_size]

        data_file = path.joinpath(f"mch_ch_r04b09_dsl_mpitask{props.comm_size}.tar.gz").name
        if props.rank == 0:
            download_and_extract(uri, base_path, data_file)
        props.comm.barrier()
    except KeyError:
        assert False, f"no data for communicator of size {props.comm_size} exists, use 1, 2 or 4"
