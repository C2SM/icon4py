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
import re

import pytest
from click.testing import CliRunner

from icon4py.pyutils.icon4pygen import main
from icon4py.testutils.utils import get_stencil_module_path


@pytest.fixture
def cli():
    return CliRunner()


STENCILS_TO_TEST = [
    ("atm_dyn_iconam", "mo_nh_diffusion_stencil_06"),
    ("atm_dyn_iconam", "mo_solve_nonhydro_stencil_27"),
    ("atm_dyn_iconam", "mo_velocity_advection_stencil_07"),
    ("atm_dyn_iconam", "mo_nh_diffusion_stencil_03"),
]


def check_gridtools_codegen(fname: str):
    patterns = {"includes": "#include <.*>", "namespaces": "using .*;"}
    with open(fname, "r") as f:
        code = f.read()
        for _, pattern in patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            assert matches


@pytest.mark.parametrize(("stencil_module", "stencil_name"), STENCILS_TO_TEST)
def test_codegen(cli, stencil_module, stencil_name):
    module_path = get_stencil_module_path(stencil_module, stencil_name)
    outpath = "."

    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, outpath])
        assert result.exit_code == 0
        check_code_was_generated(stencil_name)


def test_invalid_module_path(cli):
    module_path = get_stencil_module_path("some_module", "foo")
    outpath = "."
    result = cli.invoke(main, [module_path, outpath])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)


def test_multiple_field_operator_stencil(cli):
    module_path = get_stencil_module_path(
        "atm_dyn_iconam", "mo_velocity_advection_stencil_05"
    )
    outpath = "."
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, outpath])
        assert result.exit_code == 0


def check_code_was_generated(stencil_name):
    bindgen_header = f"{stencil_name}.h"
    gridtools_header = f"{stencil_name}.hpp"
    f90_iface = f"{stencil_name}.f90"
    assert set([bindgen_header, gridtools_header, f90_iface]).issubset(
        os.listdir(os.getcwd())
    )
    check_gridtools_codegen(gridtools_header)
