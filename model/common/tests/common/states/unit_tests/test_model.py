# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import type_alias as ta
from icon4py.model.common.states import model


def test_field_metadata_renders_only_the_set_entries() -> None:
    meta = model.FieldMetaData(standard_name="air_density", units="kg m-3")

    assert meta.dims is None
    assert meta.as_dict() == {
        "standard_name": "air_density",
        "units": "kg m-3",
        "dtype": ta.wpfloat,
    }
