# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, TypeVar

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


T = TypeVar("T")


@dataclasses.dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog
    """

    #: specific humidity [kg/kg] at cell center
    qv: fa.CellKField[ta.wpfloat]
    #: specific cloud water content [kg/kg] at cell center
    qc: fa.CellKField[ta.wpfloat]
    #: specific rain content [kg/kg] at cell center
    qr: fa.CellKField[ta.wpfloat]
    #: specific cloud ice content [kg/kg] at cell center
    qi: fa.CellKField[ta.wpfloat]
    #: specific snow content [kg/kg] at cell center
    qs: fa.CellKField[ta.wpfloat]
    #: specific graupel content [kg/kg] at cell center
    qg: fa.CellKField[ta.wpfloat]

    @classmethod
    def zero_field(cls: type[T], grid: icon_grid.IconGrid, allocator: gtx_typing.Allocator) -> T:
        tracer_dict = {
            f.name: data_alloc.zero_field(
                grid,
                dims.CellDim,
                dims.KDim,
                allocator=allocator,
                dtype=ta.wpfloat,
            )
            for f in dataclasses.fields(cls)
        }
        return cls(**tracer_dict)

    @classmethod
    def from_tracer_state(cls: type[T], tracer_state: T, allocator: gtx_typing.Allocator) -> T:
        tracer_dict = {
            f.name: data_alloc.as_field(getattr(tracer_state, f.name), allocator=allocator)
            for f in dataclasses.fields(tracer_state)
        }
        return cls(**tracer_dict)
