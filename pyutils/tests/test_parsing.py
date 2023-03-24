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

from icon4py.serialisation.parse import GranuleParser


def test_granule_parsing():
    root_dir = Path(__file__).parent
    granule = Path(f"{root_dir}/samples/granule_example.f90")
    dependencies = [
        Path(f"{root_dir}/samples/derived_types_example.f90"),
        Path(f"{root_dir}/samples/subroutine_example.f90"),
    ]
    parser = GranuleParser(granule, dependencies)
    parsed = parser.parse()

    assert list(parsed) == ["diffusion_init", "diffusion_run"]

    assert list(parsed["diffusion_init"]) == ["in"]
    assert len(parsed["diffusion_init"]["in"]) == 107
    assert parsed["diffusion_init"]["in"]["codegen_ln"] == [279]

    assert list(parsed["diffusion_run"]) == ["in", "inout", "out"]
    assert len(parsed["diffusion_run"]["in"]) == 5
    assert parsed["diffusion_run"]["in"]["codegen_ln"] == [432]

    assert len(parsed["diffusion_run"]["inout"]) == 8
    assert parsed["diffusion_run"]["inout"]["codegen_ln"] == [432, 1970]

    assert len(parsed["diffusion_run"]["out"]) == 5
    assert parsed["diffusion_run"]["out"]["codegen_ln"] == [1970]

    assert isinstance(parsed, dict)


# todo: add more tests
