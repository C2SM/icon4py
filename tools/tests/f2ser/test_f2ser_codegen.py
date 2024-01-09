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

from icon4pytools.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4pytools.f2ser.parse import GranuleParser
from icon4pytools.liskov.codegen.serialisation.generate import SerialisationCodeGenerator
from icon4pytools.liskov.codegen.shared.types import GeneratedCode


def test_deserialiser_diffusion_codegen(diffusion_granule, diffusion_granule_deps):
    parsed = GranuleParser(diffusion_granule, diffusion_granule_deps)()
    interface = ParsedGranuleDeserialiser(parsed)()
    generated = SerialisationCodeGenerator(interface)()
    assert len(generated) == 3


@pytest.fixture
def expected_no_deps_serialization_directives():
    serialization_directives = [
        GeneratedCode(
            startln=14,
            source="\n"
            '    !$ser init directory="." prefix="f2ser"\n'
            "\n"
            "    !$ser savepoint no_deps_init_in\n"
            "\n"
            "    PRINT *, 'Serializing a=a'\n"
            "\n"
            "    !$ser data a=a\n"
            "\n"
            "    PRINT *, 'Serializing b=b'\n"
            "\n"
            "    !$ser data b=b",
        ),
        GeneratedCode(
            startln=16,
            source="\n"
            "    !$ser savepoint no_deps_init_out\n"
            "\n"
            "    PRINT *, 'Serializing c=c'\n"
            "\n"
            "    !$ser data c=c\n"
            "\n"
            "    PRINT *, 'Serializing b=b'\n"
            "\n"
            "    !$ser data b=b",
        ),
        GeneratedCode(
            startln=22,
            source="\n"
            "    !$ser savepoint no_deps_run_in\n"
            "\n"
            "    PRINT *, 'Serializing a=a'\n"
            "\n"
            "    !$ser data a=a\n"
            "\n"
            "    PRINT *, 'Serializing b=b'\n"
            "\n"
            "    !$ser data b=b",
        ),
        GeneratedCode(
            startln=24,
            source="\n"
            "    !$ser savepoint no_deps_run_out\n"
            "\n"
            "    PRINT *, 'Serializing c=c'\n"
            "\n"
            "    !$ser data c=c\n"
            "\n"
            "    PRINT *, 'Serializing b=b'\n"
            "\n"
            "    !$ser data b=b",
        ),
    ]
    return serialization_directives


def test_deserialiser_directives_no_deps_codegen(
    no_deps_source_file, expected_no_deps_serialization_directives
):
    parsed = GranuleParser(no_deps_source_file)()
    interface = ParsedGranuleDeserialiser(parsed)()
    generated = SerialisationCodeGenerator(interface)()
    assert generated == expected_no_deps_serialization_directives


def test_deserialiser_directives_diffusion_codegen(
    diffusion_granule, diffusion_granule_deps, samples_path
):
    parsed = GranuleParser(diffusion_granule, diffusion_granule_deps)()
    interface = ParsedGranuleDeserialiser(parsed)()
    generated = SerialisationCodeGenerator(interface)()
    reference_savepoint = (samples_path / "expected_diffusion_granule_savepoint.f90").read_text()
    assert generated[0].source == reference_savepoint.rstrip()
