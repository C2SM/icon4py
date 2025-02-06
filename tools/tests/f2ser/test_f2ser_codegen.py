# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.tools.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.tools.f2ser.parse import GranuleParser
from icon4py.tools.liskov.codegen.serialisation.generate import SerialisationCodeGenerator
from icon4py.tools.liskov.codegen.shared.types import GeneratedCode


def test_deserialiser_diffusion_codegen(diffusion_granule, diffusion_granule_deps):
    parsed = GranuleParser(diffusion_granule, diffusion_granule_deps)()
    interface = ParsedGranuleDeserialiser(parsed)()
    generated = SerialisationCodeGenerator(interface)()
    assert len(generated) == 3


@pytest.fixture
def expected_no_deps_serialization_directives():
    serialization_directives = [
        GeneratedCode(
            startln=22,
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
            startln=24,
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
            startln=30,
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
            startln=32,
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
