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
from pathlib import Path
from pathlib import Path

from icon4py.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.f2ser.parse import GranuleParser
from icon4py.liskov.codegen.serialisation.generate import SerialisationGenerator
from icon4py.liskov.codegen.types import GeneratedCode

def test_deserialiser_diffusion_codegen(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".", prefix="test")
    interface = deserialiser()
    generator = SerialisationGenerator(interface)
    generated = generator()
    assert len(generated) == 3

@pytest.fixture
def expected_no_deps_serialization_directives():
    serialization_directives = [
        GeneratedCode(startln=12,
                      source='\n!$ser init directory="." prefix="test"\n\n!$ser savepoint no_deps_init_in\n\n!$ser data a=a\n\n!$ser data b=b'),
        GeneratedCode(startln=14,
                      source='\n!$ser savepoint no_deps_init_out\n\n!$ser data c=c\n\n!$ser data b=b'),
        GeneratedCode(startln=20,
                      source='\n!$ser savepoint no_deps_run_in\n\n!$ser data a=a\n\n!$ser data b=b'),
        GeneratedCode(startln=22,
                      source='\n!$ser savepoint no_deps_run_out\n\n!$ser data c=c\n\n!$ser data b=b')]
    return serialization_directives

def test_deserialiser_directives_no_deps_codegen(no_deps_source_file, expected_no_deps_serialization_directives):
    parser = GranuleParser(no_deps_source_file)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".", prefix="test")
    interface = deserialiser.deserialise()
    generator = SerialisationGenerator(interface)
    generated = generator()
    assert(generated == expected_no_deps_serialization_directives)

def test_deserialiser_directives_diffusion_codegen(diffusion_granule, diffusion_granule_deps,
                                                   expected_diffusion_serialization_directives):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".", prefix="test")
    interface = deserialiser.deserialise()
    generator = SerialisationGenerator(interface)
    generated = generator()
    reference_str = Path(expected_diffusion_serialization_directives).read_text()
    generated_str = ', '.join([str(elem) for elem in generated])
    generated_str = "[" + generated_str + "]\n"
    assert(generated_str == reference_str)
