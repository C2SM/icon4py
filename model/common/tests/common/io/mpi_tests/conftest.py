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


#: Seconds a single test may run before the process is taken down.
HANG_TIMEOUT_S = 900


@pytest.fixture(autouse=True)
def abort_hung_test() -> Iterator[None]:
    """Turn a hung test into a failure that carries the stacks of all threads.

    A rank blocked in an MPI call never returns to pytest, so a hung test runs the job
    into the Slurm time limit without any diagnostic. After ``HANG_TIMEOUT_S`` the
    tracebacks of all threads are written to stderr (the per-rank log) and the process
    exits, which makes srun end the whole step with an error.
    """
    faulthandler.dump_traceback_later(HANG_TIMEOUT_S, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
