# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A Fortran program passes `real(c_double)` arrays into an icon4py computation in single."""

import os
import pathlib
import subprocess
import sys

import pytest
from click.testing import CliRunner

from icon4py.tools.py2fgen import _cli, _utils


SAMPLES = pathlib.Path(__file__).parent / "fortran_samples"
LIBRARY = "cast_plugin"


@pytest.mark.single_precision_ready
def test_fortran_double_to_python_single_round_trip(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SAMPLES))
    monkeypatch.chdir(tmp_path)

    generated = CliRunner().invoke(
        _cli.main,
        [
            "boundary_cast_wrappers",
            "scale_inout,scale_in",
            LIBRARY,
            "-r",
            _utils.get_prefix_lib_path(),
        ],
    )
    assert generated.exit_code == 0, generated.output
    interface = (tmp_path / f"{LIBRARY}.f90").read_text()
    # ICON double: the boundary is double even though Python computes in single
    assert "real(c_float)" not in interface

    subprocess.run(
        [
            "gfortran",
            "-cpp",
            "-I.",
            "-Wl,-rpath=.",
            "-L.",
            f"{LIBRARY}.f90",
            str(SAMPLES / "test_boundary_casts.f90"),
            f"-l{LIBRARY}",
            "-o",
            LIBRARY,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # the embedded interpreter only sees PYTHONPATH, not pytest's sys.path additions
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, sys.path))}
    result = subprocess.run([f"./{LIBRARY}"], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout, result.stdout + result.stderr
