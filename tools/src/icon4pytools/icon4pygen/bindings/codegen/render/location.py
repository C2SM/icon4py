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
from typing import ClassVar

from icon4pytools.icon4pygen.bindings.exceptions import BindingsRenderingException


class LocationRenderer:
    type_dispatcher: ClassVar = {"Cell": "Cells", "Edge": "Edges", "Vertex": "Vertices"}

    @classmethod
    def location_type(cls, cls_name: str) -> str:
        if cls_name not in cls.type_dispatcher.keys():
            raise BindingsRenderingException(
                f"cls name {cls_name} needs to be either 'Cell', 'Edge' or 'Vertex'"
            )
        return cls.type_dispatcher[cls_name]
