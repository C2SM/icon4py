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

from pathlib import Path

import pytest

from icon4py.f2ser.exceptions import MissingDerivedTypeError, ParsingError
from icon4py.f2ser.parse import GranuleParser


def root_dir():
    return Path(__file__).parent


@pytest.fixture
def granule():
    return Path(f"{root_dir()}/samples/granule_example.f90")


def test_granule_parsing(granule):
    dependencies = [
        Path(f"{root_dir()}/samples/derived_types_example.f90"),
        Path(f"{root_dir()}/samples/subroutine_example.f90"),
    ]
    parser = GranuleParser(granule, dependencies)
    parsed = parser.parse()

    assert list(parsed) == ["diffusion_init", "diffusion_run"]

    assert list(parsed["diffusion_init"]) == ["in"]
    assert len(parsed["diffusion_init"]["in"]) == 107
    assert parsed["diffusion_init"]["in"]["codegen_lines"] == [279]

    assert list(parsed["diffusion_run"]) == ["in", "inout", "out"]
    assert len(parsed["diffusion_run"]["in"]) == 5
    assert parsed["diffusion_run"]["in"]["codegen_lines"] == [432]

    assert len(parsed["diffusion_run"]["inout"]) == 8
    assert parsed["diffusion_run"]["inout"]["codegen_lines"] == [432, 1970]

    assert len(parsed["diffusion_run"]["out"]) == 5
    assert parsed["diffusion_run"]["out"]["codegen_lines"] == [1970]

    assert isinstance(parsed, dict)


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        [Path(f"{root_dir()}/samples/subroutine_example.f90")],
    ],
)
def test_granule_parsing_missing_derived_typedef(granule, dependencies):
    parser = GranuleParser(granule, dependencies)
    with pytest.raises(
        MissingDerivedTypeError, match="Could not find type definition for TYPE"
    ):
        parser.parse()


def test_granule_parsing_no_intent():
    parser = GranuleParser(Path(f"{root_dir()}/samples/subroutine_example.f90"), [])
    with pytest.raises(ParsingError):
        parser.parse()
