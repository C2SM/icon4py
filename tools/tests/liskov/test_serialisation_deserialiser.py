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

from icon4pytools.liskov.codegen.serialisation.deserialise import (
    InitDataFactory,
    SavepointDataFactory,
)
from icon4pytools.liskov.codegen.serialisation.interface import (
    FieldSerialisationData,
    InitData,
    Metadata,
    SavepointData,
)
from icon4pytools.liskov.parsing.parse import (
    Declare,
    EndCreate,
    EndProfile,
    EndStencil,
    Imports,
    StartCreate,
    StartProfile,
    StartStencil,
)


@pytest.fixture
def parsed_dict():
    parsed = {
        "directives": [
            Imports(string="IMPORTS()", startln=0, endln=0),
            StartCreate(
                string="START CREATE()",
                startln=2,
                endln=2,
            ),
            Declare(
                string="DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; suffix=dsl)",
                startln=4,
                endln=4,
            ),
            Declare(
                string="DECLARE(vn= nproma,p_patch%nlev,p_patch%nblks_e; a=nproma,p_patch%nlev,p_patch%nblks_e; b=nproma,p_patch%nlev,p_patch%nblks_e; type=REAL(vp))",
                startln=6,
                endln=7,
            ),
            StartStencil(
                string="START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; z_nabla2_e=z_nabla2_e(:, :, 1); area_edge=p_patch%edges%area_edge(:,1); fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx; accpresent=True)",
                startln=9,
                endln=14,
            ),
            StartProfile(
                string="START PROFILE(name=apply_nabla2_to_vn_in_lateral_boundary)",
                startln=35,
                endln=35,
            ),
            EndProfile(
                string="END PROFILE()",
                startln=37,
                endln=37,
            ),
            EndStencil(
                string="END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; noprofile=True)",
                startln=38,
                endln=38,
            ),
            EndCreate(
                string="END CREATE()",
                startln=39,
                endln=39,
            ),
        ],
        "content": {
            "Imports": [{}],
            "StartCreate": [None],
            "Declare": [
                {"vn": "nproma,p_patch%nlev,p_patch%nblks_e", "suffix": "dsl"},
                {
                    "vn": "nproma,p_patch%nlev,p_patch%nblks_e",
                    "a": "nproma,p_patch%nlev,p_patch%nblks_e",
                    "b": "nproma,p_patch%nlev,p_patch%nblks_e",
                    "type": "REAL(vp)",
                },
            ],
            "StartStencil": [
                {
                    "name": "apply_nabla2_to_vn_in_lateral_boundary",
                    "z_nabla2_e": "z_nabla2_e(:,:,1)",
                    "area_edge": "p_patch%edges%area_edge(:,1)",
                    "fac_bdydiff_v": "fac_bdydiff_v",
                    "vn": "p_nh_prog%vn(:,:,1)",
                    "vertical_lower": "1",
                    "vertical_upper": "nlev",
                    "horizontal_lower": "i_startidx",
                    "horizontal_upper": "i_endidx",
                    "accpresent": "True",
                    "optional_module": "advection",
                }
            ],
            "StartProfile": [{"name": "apply_nabla2_to_vn_in_lateral_boundary"}],
            "EndProfile": [{}],
            "EndStencil": [{"name": "apply_nabla2_to_vn_in_lateral_boundary", "noprofile": "True"}],
            "EndCreate": [{}],
        },
    }
    return parsed


def test_init_data_factory(parsed_dict):
    init = InitDataFactory()(parsed_dict)
    assert isinstance(init, InitData)
    assert init.directory == "."
    assert init.prefix == "liskov-serialisation"


def test_savepoint_data_factory(parsed_dict):
    savepoints = SavepointDataFactory()(parsed_dict)
    assert len(savepoints) == 2
    assert any([isinstance(sp, SavepointData) for sp in savepoints])
    # check that unnecessary keys have been removed
    assert not any(f.variable == "optional_module" for sp in savepoints for f in sp.fields)
    assert any([isinstance(f, FieldSerialisationData) for f in savepoints[0].fields])
    assert any([isinstance(m, Metadata) for m in savepoints[0].metadata])
