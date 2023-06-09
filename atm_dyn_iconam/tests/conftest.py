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
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn
from gt4py.next.program_processors.runners.roundtrip import executor

from atm_dyn_iconam.tests.test_utils.simple_mesh import SimpleMesh


BACKENDS = {"embedded": executor, run_gtfn.name: run_gtfn}
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
