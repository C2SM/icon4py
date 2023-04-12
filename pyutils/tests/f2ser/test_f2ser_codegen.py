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

from icon4py.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.f2ser.parse import GranuleParser
from icon4py.liskov.codegen.serialisation.generate import SerialisationGenerator


def test_deserialiser_diffusion_codegen(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".", prefix="test")
    interface = deserialiser.deserialise()
    generator = SerialisationGenerator(interface)
    generated = generator()
    assert len(generated) == 3
