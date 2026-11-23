# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import faulthandler
import sys
from collections.abc import Iterator

import pytest


#: Seconds an MPI test may run before its process is taken down.
MPI_TEST_HANG_TIMEOUT_S = 900


@pytest.fixture(autouse=True)
def abort_hung_mpi_test(request: pytest.FixtureRequest) -> Iterator[None]:
    """Turn a hung MPI test into a failure that carries the stacks of all threads.

    A rank blocked in an MPI call never returns to pytest, so a hung test runs the job
    into the Slurm time limit without any diagnostic. For tests marked ``mpi``, after
    ``MPI_TEST_HANG_TIMEOUT_S`` the tracebacks of all threads are written to stderr (the
    per-rank log) and the process exits, which makes srun end the whole step with an
    error. Other tests are left alone.
    """
    if request.node.get_closest_marker("mpi") is None:
        yield
        return
    faulthandler.dump_traceback_later(MPI_TEST_HANG_TIMEOUT_S, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
