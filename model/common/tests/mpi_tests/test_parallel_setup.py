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

from icon4py.model.common.decomposition.definitions import SingleNodeRun, get_processor_properties
from icon4py.model.common.decomposition.mpi_decomposition import _get_processor_properties, init_mpi


mpi4py = pytest.importorskip("mpi4py")


@pytest.mark.mpi
def test_parallel_properties_from_comm_world():
    props = _get_processor_properties(with_mpi=True)
    assert props.rank < props.comm_size
    assert props.comm_name == "MPI_COMM_WORLD"


@pytest.mark.mpi(min_size=2)
def test_parallel_properties_from_mpi_comm():
    init_mpi()
    world = mpi4py.MPI.COMM_WORLD
    group = world.Get_group()
    pair = group.Incl([0, 1])
    comm = world.Create(pair)
    if comm != mpi4py.MPI.COMM_NULL:
        comm.Set_name("my_comm")
        props = _get_processor_properties(with_mpi=True, comm_id=comm)
        assert props.rank < props.comm_size
        assert props.comm_size == 2
        assert props.comm_name == "my_comm"


def test_single_node_properties():
    props = get_processor_properties(SingleNodeRun())
    assert props.comm_size == 1
    assert props.rank == 0
    assert props.comm_name == ""
