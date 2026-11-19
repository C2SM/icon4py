# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import dataclass_transform


@dataclass_transform(frozen_default=True, kw_only_default=True, eq_default=False)
class State:
    """
    A frozen, keyword-only dataclass of typed leaves.

    `Field` leaves are read back from the annotations; plain leaves (floats, ints,
    datetimes) are ordinary fields and are not listed. A leaf annotated `Field[Q] | None`
    is optional: `allocate` may leave it None; `fields()` skips it.

    A state is not a `Mapping`: the present leaves of an instance are read back as a
    mapping by the `fields()` free function, next to `allocate` and `copy`. States are
    identity-equal and hashable, the `eq=False` dataclass choice.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclasses.dataclass(frozen=True, eq=False, kw_only=True)(cls)


class Empty(State):
    pass
