# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar

from icon4pytools.icon4pygen.bindings.exceptions import BindingsRenderingException


class LocationRenderer:
    type_dispatcher: ClassVar[dict[str, str]] = {
        "Cell": "Cells",
        "Edge": "Edges",
        "Vertex": "Vertices",
    }

    @classmethod
    def location_type(cls, cls_name: str) -> str:
        if cls_name not in cls.type_dispatcher.keys():
            raise BindingsRenderingException(
                f"cls name {cls_name} needs to be either 'Cell', 'Edge' or 'Vertex'"
            )
        return cls.type_dispatcher[cls_name]
