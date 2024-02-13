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
import subprocess

import pytest
from click.testing import CliRunner

from icon4pytools.py2fgen.cli import main


def test_py2fgen():
    cli = CliRunner()
    module = "icon4pytools.py2fgen.wrappers.square"
    function = "square"
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module, function])
        assert result.exit_code == 0


def test_py2fgen_compilation_and_execution(samples_path):
    cli = CliRunner()
    module = "icon4pytools.py2fgen.wrappers.square"
    function = "square"

    with cli.isolated_filesystem():
        # Generate the header file, f90 interface and dynamic library
        result = cli.invoke(main, [module, function])
        assert result.exit_code == 0, "CLI execution failed"

        # Compile generated f90 interface, driver code, and dynamic library
        try:
            subprocess.run(["gfortran", "-c", "square_plugin.f90", "."], check=True)
            subprocess.run(
                [
                    "gfortran",
                    "-I.",
                    "-Wl,-rpath=.",
                    "-L.",
                    "square_plugin.f90",
                    str(samples_path / "test_square.f90"),
                    "-lsquare_plugin",
                    "-o",
                    "squarer",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compilation failed: {e}")

        # Run the compiled executable and check if it ran successfully
        try:
            fortran_result = subprocess.run(
                ["./squarer"], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")

        assert "All elements squared correctly." in fortran_result.stdout
