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

from icon4py.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4py.f2ser.interface import (
    FieldSerialisationData,
    SavepointData,
    SerialisationInterface,
)
from icon4py.f2ser.parse import GranuleParser


@pytest.fixture
def mock_parsed_granule():
    return {
        "diffusion_init": {
            "in": {
                "jg": {"typespec": "integer", "attrspec": [], "intent": ["in"]},
                "vt": {
                    "typespec": "real",
                    "kindselector": {"kind": "vp"},
                    "attrspec": [],
                    "intent": ["in"],
                    "dimension": [":", ":", ":"],
                },
                "codegen_line": 432,
            }
        },
        "diffusion_run": {
            "out": {
                "vert_idx": {
                    "typespec": "logical",
                    "kindselector": {"kind": "vp"},
                    "attrspec": [],
                    "intent": ["in"],
                    "dimension": [":", ":", ":"],
                },
                "codegen_line": 800,
            },
            "in": {
                "vn": {"typespec": "integer", "attrspec": [], "intent": ["out"]},
                "vert_idx": {
                    "typespec": "logical",
                    "kindselector": {"kind": "vp"},
                    "attrspec": [],
                    "intent": ["in"],
                    "dimension": [":", ":", ":"],
                },
                "codegen_line": 600,
            },
            "inout": {
                "vn": {"typespec": "integer", "attrspec": [], "intent": ["out"]},
                "vert_idx": {
                    "typespec": "logical",
                    "kindselector": {"kind": "vp"},
                    "attrspec": [],
                    "intent": ["in"],
                    "dimension": [":", ":", ":"],
                },
            },
        },
    }


def test_deserialiser_mock(mock_parsed_granule):
    deserialiser = ParsedGranuleDeserialiser(mock_parsed_granule, directory=".")
    interface = deserialiser.deserialise()
    assert isinstance(interface, SerialisationInterface)
    assert len(interface.savepoint) == 3
    assert all([isinstance(s, SavepointData) for s in interface.savepoint])
    assert all(
        [
            isinstance(f, FieldSerialisationData)
            for s in interface.savepoint
            for f in s.fields
        ]
    )


def test_deserialiser_diffusion_granule(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser.parse()
    deserialiser = ParsedGranuleDeserialiser(parsed, directory=".")
    interface = deserialiser.deserialise()
    assert len(interface.savepoint) == 3
