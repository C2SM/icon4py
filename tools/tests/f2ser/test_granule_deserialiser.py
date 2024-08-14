# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4pytools.f2ser.deserialise import ParsedGranuleDeserialiser
from icon4pytools.f2ser.parse import CodegenContext, GranuleParser, ParsedGranule
from icon4pytools.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    SavepointData,
    SerialisationCodeInterface,
)


@pytest.fixture
def mock_parsed_granule():
    return ParsedGranule(
        subroutines={
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
                    "codegen_ctx": CodegenContext(432, 450, 600),
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
                    "codegen_ctx": CodegenContext(800, 850, 1000),
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
                    "codegen_ctx": CodegenContext(600, 690, 750),
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
        },
        last_import_ln=59,
    )


def test_deserialiser_mock(mock_parsed_granule):
    deserialiser = ParsedGranuleDeserialiser(mock_parsed_granule)
    interface = deserialiser()
    assert isinstance(interface, SerialisationCodeInterface)
    assert len(interface.Savepoint) == 3
    assert all([isinstance(s, SavepointData) for s in interface.Savepoint])
    assert all(
        [isinstance(f, FieldSerialisationData) for s in interface.Savepoint for f in s.fields]
    )


def test_deserialiser_diffusion_granule(diffusion_granule, diffusion_granule_deps):
    parser = GranuleParser(diffusion_granule, diffusion_granule_deps)
    parsed = parser()
    deserialiser = ParsedGranuleDeserialiser(parsed)
    interface = deserialiser()
    assert len(interface.Savepoint) == 3
