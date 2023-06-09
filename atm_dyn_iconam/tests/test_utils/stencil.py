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


def _test_validation(self, mesh, backend, input_data):
    reference_outputs = self.reference(
        mesh, **{k: np.array(v) for k, v in input_data.items()}
    )
    self.PROGRAM.with_backend(backend)(
        **input_data,
        offset_provider=mesh.get_offset_provider(),
    )
    for name in self.OUTPUTS:
        assert np.allclose(
            input_data[name], reference_outputs[name]
        ), f"Validation failed for '{name}'"


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
    def __init_subclass__(cls, **kwargs):
        # Add two methods for verification and benchmarking. In order to have names that
        # reflect the name of the test we do this dynamically here instead of using regular
        # inheritance.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_validation)
        setattr(cls, f"bench_{cls.__name__}", _bench_execution)
