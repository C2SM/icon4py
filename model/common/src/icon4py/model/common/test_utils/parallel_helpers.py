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

from icon4py.model.common.decomposition.definitions import ProcessProperties, get_runtype
from icon4py.model.common.decomposition.mpi_decomposition import get_multinode_properties


def check_comm_size(props: ProcessProperties, sizes=(1, 2, 4)):
    if props.comm_size not in sizes:
        pytest.xfail(f"wrong comm size: {props.comm_size}: test only works for comm-sizes: {sizes}")


@pytest.fixture(scope="session")
def processor_props(request):
    runtype = get_runtype(with_mpi=True)
    yield get_multinode_properties(runtype)
