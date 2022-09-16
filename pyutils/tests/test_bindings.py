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
from pathlib import Path

import pytest

import icon4py.atm_dyn_iconam
from icon4py.bindings.workflow import CppBindGen, PyBindGen
from icon4py.pyutils.icon4pygen import get_stencil_metadata


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

    for fencil in fencils:
        metadata = get_stencil_metadata(fencil)
        CppBindGen(metadata)(cppbindgen_path)
        PyBindGen(metadata)(output_path)
        compare_files(cppbindgen_path, output_path, fencil)


def compare_files(cppbindgen_dir: Path, pybindgen_dir: Path, fencil: str):
    fname = f"{fencil.split(':')[1]}"

    cppgen_name = cppbindgen_dir / Path(f"build/generated/{fname}.h")
    pygen_name = pybindgen_dir / Path(f"{fname}.h")

    with open(cppgen_name, "r+") as cpp:
        cpp_f = cpp.read()
        cpp_f = "".join(cpp_f.split())
        with open(pygen_name, "r+") as py:
            py_f = py.read()
            py_f = "".join(py_f.split())
            assert cpp_f == py_f
