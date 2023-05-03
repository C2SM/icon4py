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

import pytest

from icon4py.f2ser.exceptions import MissingDerivedTypeError, ParsingError
from icon4py.f2ser.parse import CodegenContext, GranuleParser


def test_granule_parsing(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser()

    assert list(parsed) == ["diffusion_init", "diffusion_run"]

    assert list(parsed["diffusion_init"]) == ["in"]
    assert len(parsed["diffusion_init"]["in"]) == 107
    assert parsed["diffusion_init"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=190, last_declaration_ln=280, end_subroutine_ln=401
    )

    assert list(parsed["diffusion_run"]) == ["in", "inout", "out"]
    assert len(parsed["diffusion_run"]["in"]) == 5
    assert parsed["diffusion_run"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=417, last_declaration_ln=492, end_subroutine_ln=1965
    )

    assert len(parsed["diffusion_run"]["inout"]) == 8

    assert len(parsed["diffusion_run"]["out"]) == 5
    assert parsed["diffusion_run"]["out"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=417, last_declaration_ln=492, end_subroutine_ln=1965
    )

    assert isinstance(parsed, dict)


def test_granule_parsing_missing_derived_typedef(diffusion_granule, samples_path):
    dependencies = [samples_path / "subroutine_example.f90"]
    parser = GranuleParser(diffusion_granule, dependencies)
    with pytest.raises(
        MissingDerivedTypeError, match="Could not find type definition for TYPE"
    ):
        parser()


def test_granule_parsing_no_intent(samples_path):
    parser = GranuleParser(samples_path / "subroutine_example.f90", [])
    with pytest.raises(ParsingError):
        parser()
