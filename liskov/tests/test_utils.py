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
from pathlib import Path

import pytest
from functional.ffront.decorator import Program

from icon4py.liskov.parsing.exceptions import UnknownStencilError
from icon4py.liskov.parsing.types import Create, Imports, TypedDirective
from icon4py.liskov.parsing.utils import StencilCollector, extract_directive


def test_extract_directive():
    directives = [
        TypedDirective(
            string="!$DSL CREATE()", startln=1, endln=1, directive_type=Create()
        ),
        TypedDirective(
            string="!$DSL IMPORT()", startln=2, endln=2, directive_type=Imports()
        ),
    ]

    # Test that only the expected directive is extracted.
    assert extract_directive(directives, Create) == [directives[0]]
    assert extract_directive(directives, Imports) == [directives[1]]


def test_stencil_collector():
    name = "mo_nh_diffusion_stencil_06"
    collector = StencilCollector(name)
    assert isinstance(collector.fvprog, Program)


def test_stencil_collector_invalid_module():
    name = "non_existent_module"
    collector = StencilCollector(name)
    with pytest.raises(UnknownStencilError):
        collector.fvprog


def test_stencil_collector_invalid_member():
    from icon4py.atm_dyn_iconam import mo_nh_diffusion_stencil_01

    module_path = Path(mo_nh_diffusion_stencil_01.__file__)
    parents = module_path.parents[0]

    collector = StencilCollector("foo")

    path = os.path.join(parents, "foo.py")
    with open(path, "w") as f:
        f.write("")

    with pytest.raises(UnknownStencilError):
        collector.fvprog

    os.remove(path)
