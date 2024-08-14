# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4pytools.f2ser.exceptions import MissingDerivedTypeError, ParsingError
from icon4pytools.f2ser.parse import CodegenContext, GranuleParser


def test_granule_parsing(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed_granule = parser()

    subroutines = parsed_granule.subroutines

    assert list(subroutines) == ["diffusion_init", "diffusion_run"]

    assert list(subroutines["diffusion_init"]) == ["in"]
    assert len(subroutines["diffusion_init"]["in"]) == 107
    assert subroutines["diffusion_init"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=190, last_declaration_ln=280, end_subroutine_ln=401
    )

    assert list(subroutines["diffusion_run"]) == ["in", "inout", "out"]
    assert len(subroutines["diffusion_run"]["in"]) == 5
    assert subroutines["diffusion_run"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=417, last_declaration_ln=492, end_subroutine_ln=1965
    )

    assert len(subroutines["diffusion_run"]["inout"]) == 8

    assert len(subroutines["diffusion_run"]["out"]) == 5
    assert subroutines["diffusion_run"]["out"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=417, last_declaration_ln=492, end_subroutine_ln=1965
    )

    assert isinstance(subroutines, dict)
    assert parsed_granule.last_import_ln == 60


def test_granule_parsing_missing_derived_typedef(diffusion_granule, samples_path):
    dependencies = [samples_path / "subroutine_example.f90"]
    parser = GranuleParser(diffusion_granule, dependencies)
    with pytest.raises(MissingDerivedTypeError, match="Could not find type definition for TYPE"):
        parser()


def test_granule_parsing_no_intent(samples_path):
    parser = GranuleParser(samples_path / "subroutine_example.f90", [])
    with pytest.raises(ParsingError):
        parser()


def test_multiline_declaration_parsing(samples_path):
    parser = GranuleParser(samples_path / "multiline_example.f90", [])
    parsed_granule = parser()
    subroutines = parsed_granule.subroutines
    assert list(subroutines) == ["graupel_init", "graupel_run"]
    assert subroutines["graupel_init"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=121, last_declaration_ln=145, end_subroutine_ln=231
    )
    assert subroutines["graupel_run"]["in"]["codegen_ctx"] == CodegenContext(
        first_declaration_ln=254, last_declaration_ln=301, end_subroutine_ln=419
    )
