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
import numpy as np
import pytest
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn
from gt4py.next.program_processors.runners.roundtrip import executor

from atm_dyn_iconam.tests.test_utils.simple_mesh import SimpleMesh


BACKENDS = [executor, run_gtfn]
BACKEND_NAMES = ["embedded", run_gtfn.name]

MESHES = [SimpleMesh()]
MESH_NAMES = ["simple_mesh"]


@pytest.fixture(params=MESHES, ids=MESH_NAMES)
def mesh(request):
    return request.param


@pytest.fixture(params=BACKENDS, ids=BACKEND_NAMES)
def backend(request):
    return request.param


def _test_validation(self, mesh, backend, input_data):
    reference_outputs = self.reference(  # noqa: F841
        mesh, **{k: np.array(v) for k, v in input_data.items()}
    )
    self.PROGRAM.with_backend(backend)(
        **input_data,
        offset_provider=mesh.get_offset_provider(),
    )
    for name in self.OUTPUTS:
        exec(
            f"assert np.allclose(input_data['{name}'][{self.OUT_INDEX}], reference_outputs['{name}'][{self.OUT_INDEX}])"
        )


def _bench_execution(self, pytestconfig, mesh, backend, input_data, benchmark):
    if pytestconfig.getoption(
        "--benchmark-disable"
    ):  # skipping as otherwise program calls are duplicated in tests.
        pytest.skip("Test skipped due to 'benchmark-disable' option.")
    else:
        benchmark(
            self.PROGRAM.with_backend(backend),
            **input_data,
            offset_provider=mesh.get_offset_provider(),
        )


class StencilTest:
    # The out index is specifiable in the cases in which the output domain needs to be specified due to offsets.
    OUT_INDEX = ":"

    def __init_subclass__(cls, **kwargs):
        # The subclass will have two methods which are registered as test functions by pytest.
        # This allows both the validation test and the benchmark test to be run.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_validation)
        setattr(cls, f"bench_{cls.__name__}", _bench_execution)
