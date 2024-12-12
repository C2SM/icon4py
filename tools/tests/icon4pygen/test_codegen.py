# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pkgutil
import re
import traceback
from importlib import reload

import icon4py.model.atmosphere.diffusion.stencils as diffusion
import icon4py.model.atmosphere.dycore.stencils as dycore
import icon4py.model.common.interpolation.stencils as intp
import icon4py.model.common.type_alias as type_alias
import pytest
from gt4py.next.ffront.fbuiltins import float32, float64

from icon4pytools.icon4pygen.cli import main

from .conftest import get_stencil_module_path


DYCORE_PKG = "atmosphere.dycore.stencils"
INTERPOLATION_PKG = "common.interpolation.stencils"
DIFFUSION_PKG = "atmosphere.diffusion.stencils"

LEVELS_PER_THREAD = "1"
BLOCK_SIZE = "128"
OUTPATH = "."


def dycore_fencils() -> list[tuple[str, str]]:
    return _fencils(dycore.__file__, DYCORE_PKG)


def interpolation_fencils() -> list[tuple[str, str]]:
    return _fencils(intp.__file__, INTERPOLATION_PKG)


def diffusion_fencils() -> list[tuple[str, str]]:
    return _fencils(diffusion.__file__, DIFFUSION_PKG)


def _fencils(module_name, package_name) -> list[tuple[str, str]]:
    pkgpath = os.path.dirname(module_name)
    stencils = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    fencils = [(package_name, stencil) for stencil in stencils]
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
        f"wrap_run_and_verify_{stencil_name}",
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
    patterns = ["#include <.*>", "using .*;", f"inline\\s+auto\\s+{stencil_name}"]
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


# TODO: (samkellerhals) add temporaries codegen here once all work.
@pytest.mark.parametrize(
    ("stencil_module", "stencil_name"),
    dycore_fencils(),
)
@pytest.mark.parametrize("flags", [()], ids=["normal"])
def test_codegen(cli, stencil_module, stencil_name, flags, test_temp_dir) -> None:
    module_path = get_stencil_module_path(stencil_module, stencil_name)
    with cli.isolated_filesystem(temp_dir=test_temp_dir):
        cli_args = [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, OUTPATH, *flags]
        result = cli.invoke(main, cli_args)
        assert (
            result.exit_code == 0
        ), f"Codegen failed with error:\n{''.join(traceback.format_exception(*result.exc_info))}"
        check_code_was_generated(stencil_name)


def test_invalid_module_path(cli) -> None:
    module_path = get_stencil_module_path("some_module", "foo")
    result = cli.invoke(main, [module_path, BLOCK_SIZE, LEVELS_PER_THREAD, OUTPATH])
    assert result.exit_code == 1
    assert isinstance(result.exception, ModuleNotFoundError)


def test_mixed_precision_option(cli) -> None:
    module_path = get_stencil_module_path("some_module", "foo")
    cli.invoke(
        main, [module_path, "--enable-mixed-precision", BLOCK_SIZE, LEVELS_PER_THREAD, OUTPATH]
    )
    reload(type_alias)
    assert os.environ.get("FLOAT_PRECISION") == "mixed"
    assert (type_alias.vpfloat == float32) and (type_alias.wpfloat == float64)
