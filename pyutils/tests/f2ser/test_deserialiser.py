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
from icon4py.f2ser.interface import SerialisationInterface


mock_granule = {
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
            "codegen_lines": [432],
        }
    },
    "diffusion_run": {
        "out": {
            "vn": {"typespec": "integer", "attrspec": [], "intent": ["out"]},
            "vert_idx": {
                "typespec": "logical",
                "kindselector": {"kind": "vp"},
                "attrspec": [],
                "intent": ["in"],
                "dimension": [":", ":", ":"],
            },
            "codegen_lines": [800],
        }
    },
}


def test_deserialiser():
    deserialiser = ParsedGranuleDeserialiser(mock_granule)
    interface = deserialiser.deserialise()
    assert isinstance(interface, SerialisationInterface)
