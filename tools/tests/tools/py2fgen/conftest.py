# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pathlib
import sys

import pytest


@pytest.fixture
def samples_path():
    return pathlib.Path(__file__).parent / "fortran_samples"


@pytest.fixture
def fortran_subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward parent sys.path through PYTHONPATH for the test's duration.

    Compiled .so files spawn a fresh Python interpreter (via CFFI's embedded
    init) that inherits the subprocess's env vars but not pytest's
    rootdir-induced sys.path additions. Mirroring sys.path through PYTHONPATH
    lets the embedded interpreter resolve test fixtures like
    ``tests.tools.py2fgen.wrappers.simple``.

    Request this fixture from any test that spawns a subprocess linking the
    compiled .so. Tests that only run the CLI in-process via
    ``cli.invoke(...)`` don't need it.
    """
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [*sys.path, os.environ.get("PYTHONPATH", "")])),
    )
