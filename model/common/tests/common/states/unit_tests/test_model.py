# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.states import model


def test_field_metadata_accepts_kind() -> None:
    meta = model.FieldMetaData(
        standard_name="tend_temperature", units="K s-1", kind=model.FieldKind.TENDENCY
    )
    assert meta.kind == model.FieldKind.TENDENCY
    # unset optional entries read as None and are left out of the rendered attrs
    assert meta.dims is None
    assert "dims" not in meta.as_dict()
