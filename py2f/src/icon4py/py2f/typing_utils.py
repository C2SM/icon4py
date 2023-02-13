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

from functional.common import Dimension
from functional.type_system.type_specifications import ScalarType
from functional.type_system.type_translation import from_type_hint


def parse_annotation(annotation) -> tuple[list[Dimension], ScalarType]:
    type_spec = from_type_hint(annotation)
    if isinstance(type_spec, ScalarType):
        dtype = type_spec.kind
        dims = []
    else:
        dtype = type_spec.dtype.kind
        dims = type_spec.dims
    return dims, dtype
