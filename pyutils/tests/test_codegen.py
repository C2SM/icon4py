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
OUTPATH = "."


@pytest.fixture
def cli():
    return CliRunner()


def atm_dyn_iconam_fencils() -> list[tuple[str, str]]:
    pkgpath = os.path.dirname(icon4py.atm_dyn_iconam.__file__)
    stencils = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    fencils = [("atm_dyn_iconam", stencil) for stencil in stencils]
    return fencils


def check_cpp_codegen(fname: str) -> None:
    stencil_name = fname.replace(".cpp", "")
    patterns = [
        "#include <.*>",
        "using .*;",
        "template <.*>",
        f"class {stencil_name}",
        "GpuTriMesh",
        "dim3",
        "static void setup",
        "void run",
        "copy_pointers",
        f"void run_{stencil_name}",
        f"bool verify_{stencil_name}",
        f"void run_and_verify_{stencil_name}",
        f"void setup_{stencil_name}",
        f"void free_{stencil_name}",
    ]
    check_for_matches(fname, patterns)


def check_fortran_codegen(fname: str) -> None:
    stencil_name = fname.replace(".f90", "")
    patterns = [
        f"module {stencil_name}",
        "use, intrinsic :: iso_c_binding",
        f"run_{stencil_name}",
        f"run_and_verify_{stencil_name}",
        f"setup_{stencil_name}",
        f"free_{stencil_name}",
        f"wrap_run_{stencil_name}",
    ]
    check_for_matches(fname, patterns)


def check_header_codegen(fname: str) -> None:
    stencil_name = fname.replace(".h", "")
    patterns = [
        '#include ".*"',
        f"void run_{stencil_name}",
        f"bool verify_{stencil_name}",
        f"void run_and_verify_{stencil_name}",
        f"void setup_{stencil_name}",
        f"void free_{stencil_name}",
    ]
    check_for_matches(fname, patterns)


def check_gridtools_codegen(fname: str) -> None:
    stencil_name = fname.replace(".hpp", "")
    patterns = ["#include <.*>", "using .*;", f"inline auto {stencil_name}"]
    check_for_matches(fname, patterns)


def check_for_matches(fname: str, patterns: list[str]) -> None:
    with open(fname, "r") as f:
        code = f.read()
        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            assert matches


def check_code_was_generated(stencil_name: str) -> None:
    check_gridtools_codegen(f"{stencil_name}.hpp")
    check_fortran_codegen(f"{stencil_name}.f90")
    check_header_codegen(f"{stencil_name}.h")
    check_cpp_codegen(f"{stencil_name}.cpp")


@pytest.mark.skip("raises exception due to dims in offset provider")
@pytest.mark.parametrize(("stencil_module", "stencil_name"), atm_dyn_iconam_fencils())
def test_codegen_atm_dyn_iconam(cli, stencil_module, stencil_name) -> None:
    module_path = get_stencil_module_path(stencil_module, stencil_name)
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, OUTPATH])
        assert result.exit_code == 0
        check_code_was_generated(stencil_name)


def test_invalid_module_path(cli) -> None:
    module_path = get_stencil_module_path("some_module", "foo")
    result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, OUTPATH])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)


@pytest.mark.skip("raises exception due to dims in offset provider")
def test_codegen_mo_nh_diffusion_stencil_14(cli) -> None:
    stencil_name = "calculate_nabla2_of_theta"
    module_path = get_stencil_module_path("atm_dyn_iconam", stencil_name)
    with cli.isolated_filesystem():
        result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, IS_GLOBAL, OUTPATH])
        assert result.exit_code == 0
        check_code_was_generated(stencil_name)
