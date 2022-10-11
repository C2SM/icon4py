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
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Dimension, Field

from icon4py.bindings.entities import Offset, chain_from_str
from icon4py.bindings.exceptions import BindingsTypeConsistencyException
from icon4py.bindings.workflow import PyBindGen
from icon4py.common.dimension import EdgeDim
from icon4py.pyutils.metadata import get_stencil_info


def test_invalid_offset():
    with pytest.raises(BindingsTypeConsistencyException):
        Offset(chain="E2X")


def test_invalid_field_in_program():
    ZDim = Dimension("Z")

    @field_operator
    def bad_stencil(
        z: Field[[ZDim], float],
    ) -> Field[[ZDim], float]:
        return z

    @program
    def bad_program(
        z: Field[[ZDim], float],
    ):
        bad_stencil(z, out=z)

    with pytest.raises(BindingsTypeConsistencyException):
        stencil_info = get_stencil_info(bad_program)
        PyBindGen(stencil_info, 128, 1)


def test_chain_from_str():
    with pytest.raises(BindingsTypeConsistencyException):
        chain_from_str("E2Z")


def test_non_sparse_field_neighbors():
    @field_operator
    def bad_stencil(
        a: Field[[EdgeDim], float],
    ) -> Field[[EdgeDim], float]:
        return a

    @program
    def bad_program(
        a: Field[[EdgeDim], float],
    ):
        bad_stencil(a, out=a)

    with pytest.raises(BindingsTypeConsistencyException):
        stencil_info = get_stencil_info(bad_program)
        bindgen = PyBindGen(stencil_info, 128, 1)
        [field.get_num_neighbors() for field in bindgen.fields]
