# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
