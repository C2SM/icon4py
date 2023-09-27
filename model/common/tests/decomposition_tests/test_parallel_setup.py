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


import mpi4py
import pytest

from icon4py.model.common.decomposition.parallel_setup import get_processor_properties, init_mpi


mpi4py.rc.initialize = False
from mpi4py import MPI  # noqa : E402 # module level need to set mpi4py config before import


@pytest.mark.mpi
def test_parallel_properties_from_comm_world():
    props = get_processor_properties(with_mpi=True)
    assert props.rank < props.comm_size
    assert props.comm_name == "MPI_COMM_WORLD"


@pytest.mark.mpi(min_size=2)
def test_parallel_properties_from_mpi_comm():
    init_mpi()
    world = MPI.COMM_WORLD
    group = world.Get_group()
    pair = group.Incl([0, 1])
    comm = world.Create(pair)
    if comm != MPI.COMM_NULL:
        comm.Set_name("my_comm")
        props = get_processor_properties(with_mpi=True, comm_id=comm)
        assert props.rank < props.comm_size
        assert props.comm_size == 2
        assert props.comm_name == "my_comm"


def test_single_node_properties():
    props = get_processor_properties()
    assert props.comm_size == 1
    assert props.rank == 0
    assert props.comm_name == ""
