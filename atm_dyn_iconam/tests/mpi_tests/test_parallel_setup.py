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

import pathlib

import pytest

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.driver.io_utils import SerializationType, read_decomp_info
from icon4py.driver.parallel_setup import (
    DecompositionInfo,
    get_processor_properties,
)


"""
running tests with mpi:

mpirun -np 2 python -m pytest --with-mpi tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""


# TODO [magdalena] fix this
@pytest.mark.mpi
def test_processor_properties_from_comm_world(mpi):
    props = get_processor_properties()
    assert props.rank < mpi.COMM_WORLD.Get_size()
    assert props.comm_name == mpi.COMM_WORLD.Get_name()


# TODO s [magdalena] extract fixture, more useful asserts..., fix run parallel (as is will not work on second node...)
@pytest.mark.parametrize(
    ("dim, owned, total"),
    ((CellDim, 10448, 10611), (EdgeDim, 15820, 16065), (VertexDim, 5373, 5455)),
)
def test_decomposition_info_masked(dim, owned, total):
    path = pathlib.Path(
        "/home/magdalena/data/exclaim/dycore/mch_ch_r04b09_dsl/mpitasks2/mch_ch_r04b09_dsl/ser_data"
    )
    props = get_processor_properties()
    decomposition_info = read_decomp_info(path, props, SerializationType.SB)
    owned_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.ALL
    )
    assert owned_indices.shape[0] == total

    owned_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == owned

    halo_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.HALO
    )
    assert halo_indices.shape[0] == total - owned
