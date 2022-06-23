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

from icon4py.pyutils.exceptions import MultipleFieldOperatorException
from icon4py.pyutils.icon4pygen import main
from icon4py.testutils.utils import get_stencil_module_path


@pytest.fixture
def cli():
    return CliRunner()


CODEGEN_PATTERNS = {"includes": "#include <.*>", "namespaces": "using .*;"}
STENCILS_TO_TEST = [
    ("atm_dyn_iconam", "mo_nh_diffusion_stencil_06"),
    ("atm_dyn_iconam", "mo_solve_nonhydro_stencil_27"),
    ("atm_dyn_iconam", "mo_velocity_advection_stencil_07"),
]


@pytest.mark.parametrize(("stencil_module", "stencil_name"), STENCILS_TO_TEST)
def test_codegen(cli, stencil_module, stencil_name):
    module_path = get_stencil_module_path(stencil_module, stencil_name)
    result = cli.invoke(main, [module_path])
    assert result.exit_code == 0
    for _, pattern in CODEGEN_PATTERNS.items():
        matches = re.findall(pattern, result.output, re.MULTILINE)
        assert matches


@pytest.mark.parametrize(("stencil_module", "stencil_name"), STENCILS_TO_TEST)
def test_metadatagen(cli, stencil_module, stencil_name):
    fname = f"{stencil_name}.dat"
    module_path = get_stencil_module_path(stencil_module, stencil_name)

    with cli.isolated_filesystem():
        result = cli.invoke(
            main, [module_path, "--output-metadata", f"{stencil_name}.dat"]
        )
        assert result.exit_code == 0
        assert fname in os.listdir(os.getcwd()) and os.path.getsize(fname) > 0


def test_invalid_module_path(cli):
    module_path = get_stencil_module_path("some_module", "foo")
    result = cli.invoke(main, [module_path])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)


def test_multiple_field_operator_stencil(cli):
    module_path = get_stencil_module_path(
        "atm_dyn_iconam", "mo_velocity_advection_stencil_05"
    )
    result = cli.invoke(main, [module_path])
    assert result.exit_code == 1
    assert isinstance(result.exception, MultipleFieldOperatorException)
