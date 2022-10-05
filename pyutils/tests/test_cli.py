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

import os
import pkgutil
import re

import pytest
from click.testing import CliRunner

import icon4py.atm_dyn_iconam
from icon4py.pyutils.icon4pygen import main
from icon4py.testutils.utils import get_stencil_module_path


LEVELS_PER_THREAD = "1"
BLOCK_SIZE = "128"


@pytest.fixture
def cli():
    return CliRunner()


def atm_dyn_iconam_fencils():
    pkgpath = os.path.dirname(icon4py.atm_dyn_iconam.__file__)
    stencils = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    fencils = [("atm_dyn_iconam", stencil) for stencil in stencils]
    return fencils


# add check for cpp files

# add check for fortran files


def check_gridtools_codegen(fname: str):
    patterns = {"includes": "#include <.*>", "namespaces": "using .*;"}  # todo: extend
    with open(fname, "r") as f:
        code = f.read()
        for _, pattern in patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            assert matches


@pytest.mark.parametrize(("stencil_module", "stencil_name"), atm_dyn_iconam_fencils())
def test_codegen_atm_dyn_iconam(cli, stencil_module, stencil_name):
    module_path = get_stencil_module_path(stencil_module, stencil_name)
    outpath = "."

    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, outpath])
        assert result.exit_code == 0
        check_code_was_generated(stencil_name)


def test_invalid_module_path(cli):
    module_path = get_stencil_module_path("some_module", "foo")
    outpath = "."
    result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, outpath])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)


def check_code_was_generated(stencil_name: str):
    cpp_header = f"{stencil_name}.h"
    gridtools_header = f"{stencil_name}.hpp"
    f90_iface = f"{stencil_name}.f90"
    cpp_def = f"{stencil_name}.cpp"
    assert set([cpp_header, gridtools_header, f90_iface, cpp_def]).issubset(
        os.listdir(os.getcwd())
    )
    check_gridtools_codegen(gridtools_header)
