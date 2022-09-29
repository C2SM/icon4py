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
import random
from pathlib import Path

import pytest

import icon4py.atm_dyn_iconam
from icon4py.bindings.workflow import CppBindGen, PyBindGen
from icon4py.pyutils.icon4pygen import get_stencil_info


@pytest.fixture
def fencils():
    pkgpath = os.path.dirname(icon4py.atm_dyn_iconam.__file__)
    stencils = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
    fencils = [f"icon4py.atm_dyn_iconam.{stencil}:{stencil}" for stencil in stencils]
    return fencils


@pytest.fixture
def cppbindgen_path():
    return Path(os.path.dirname(icon4py.bindings.__file__)).parent / Path("cppbindgen")


def test_pybindgen_against_cppbindgen(fencils, cppbindgen_path):
    output_path = Path(os.getcwd()) / Path("tmp")

    # select random sample of fencils (processing time too large otherwise)
    selected_fencils = random.choices(fencils, k=10)
    levels_per_thread = 1
    block_size = 128

    for fencil in selected_fencils:
        metadata = get_stencil_info(fencil)
        CppBindGen(metadata)(cppbindgen_path)
        PyBindGen(metadata, levels_per_thread, block_size)(output_path)
        compare_files(cppbindgen_path, output_path, fencil, "h")
        compare_files(cppbindgen_path, output_path, fencil, "f90")
        compare_files(cppbindgen_path, output_path, fencil, "cpp")


def compare_files(
    cppbindgen_dir: Path, pybindgen_dir: Path, fencil: str, extension: str
):
    fname = f"{fencil.split(':')[1]}"

    cppgen_name = cppbindgen_dir / Path(f"build/generated/{fname}.{extension}")
    pygen_name = pybindgen_dir / Path(f"{fname}.{extension}")

    with open(cppgen_name, "r+") as cpp:
        cpp_f = cpp.read()
        cpp_f = "".join(cpp_f.split())
        with open(pygen_name, "r+") as py:
            py_f = py.read()
            py_f = "".join(py_f.split())
            assert cpp_f == py_f
