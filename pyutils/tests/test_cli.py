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

import pytest
from click.testing import CliRunner

from icon4py.pyutils.icon4pygen import main


def get_module_path(module, stencil_name) -> str:
    return f"icon4py.{module}.{stencil_name}:{stencil_name}"


@pytest.fixture
def cli():
    return CliRunner()


CPP_HEADERS = """#include <gridtools/fn/unstructured.hpp>

namespace generated {
using namespace gridtools;
using namespace fn;
using namespace literals;"""


def test_codegen_single_fo(cli):
    stencil_name = "mo_nh_diffusion_stencil_06"
    module_path = get_module_path("atm_dyn_iconam", stencil_name)
    result = cli.invoke(main, [module_path])
    assert result.exit_code == 0
    assert CPP_HEADERS in result.output


# TODO: mark parametrise with different stencils.
def test_codegen_multiple_fo(cli):
    stencil_name = "mo_velocity_advection_stencil_05"
    module_path = get_module_path("atm_dyn_iconam", stencil_name)
    result = cli.invoke(main, [module_path])
    assert result.exit_code == 0
    assert CPP_HEADERS in result.output


# TODO: mark parametrise with different stencils.
def test_metadatagen_multiple_fo(cli):
    stencil_name = "mo_velocity_advection_stencil_05"
    fname = f"{stencil_name}.dat"
    module_path = get_module_path("atm_dyn_iconam", stencil_name)

    with cli.isolated_filesystem():
        result = cli.invoke(
            main, [module_path, "--output-metadata", f"{stencil_name}.dat"]
        )
        assert result.exit_code == 0
        assert fname in os.listdir(os.getcwd())


def test_metadatagen_single_fo(cli):
    stencil_name = "mo_nh_diffusion_stencil_06"
    fname = f"{stencil_name}.dat"
    module_path = get_module_path("atm_dyn_iconam", stencil_name)
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, "--output-metadata", fname])
        assert result.exit_code == 0
        assert fname in os.listdir(os.getcwd())
