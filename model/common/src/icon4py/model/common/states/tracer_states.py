# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.framework import states
from icon4py.model.common.states.data import COMMON_TRACER_CF_ATTRIBUTES
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


class TracerField(NamedTuple):
    name: str
    field: fa.CellKField[ta.wpfloat]


def _tracer_list_index(name: str) -> int:
    """The tracer's Fortran list index; every tracer must declare one."""
    index = COMMON_TRACER_CF_ATTRIBUTES[name].icon_var_list_index
    if index is None:
        raise ValueError(f"tracer '{name}' must declare icon_var_list_index")
    return index


#: Tracer names sorted by their Fortran index (QV=0, QC=1, QI=2, QR=3, QS=4, QG=5).
_TRACER_FIELDS: tuple[str, ...] = tuple(sorted(COMMON_TRACER_CF_ATTRIBUTES, key=_tracer_list_index))


# TODO(jcanton): temporary, until the physics and the IO run on the typed states
TracerConfig = states.TracerConfig


@dataclasses.dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog.

    Kept out of `PrognosticState` on purpose: tracers are advanced once per model time
    step, not once per dynamics substep, so their two time levels are swapped at a
    different rate than the dynamical ones (nnow_rcf/nnew_rcf vs nnow/nnew in ICON).
    """

    #: specific humidity [kg/kg] at cell center
    qv: fa.CellKField[ta.wpfloat] | None = None
    #: specific cloud water content [kg/kg] at cell center
    qc: fa.CellKField[ta.wpfloat] | None = None
    #: specific cloud ice content [kg/kg] at cell center
    qi: fa.CellKField[ta.wpfloat] | None = None
    #: specific rain content [kg/kg] at cell center
    qr: fa.CellKField[ta.wpfloat] | None = None
    #: specific snow content [kg/kg] at cell center
    qs: fa.CellKField[ta.wpfloat] | None = None
    #: specific graupel content [kg/kg] at cell center
    qg: fa.CellKField[ta.wpfloat] | None = None

    def active_fields(self) -> Iterator[TracerField]:
        """Yield a ``TracerField`` for each non-``None`` tracer field."""
        for name in _TRACER_FIELDS:
            field = getattr(self, name)
            if field is not None:
                yield TracerField(name=name, field=field)

    def copy(self, allocator: gtx_typing.Allocator | None = None) -> TracerState:
        """A new state with a copy of each active field, for the other time level."""
        return TracerState(
            **{
                tracer.name: data_alloc.reallocate(tracer.field, allocator=allocator)
                for tracer in self.active_fields()
            }
        )


def initialize_tracer_state(
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator | None,
    tracer_config: TracerConfig | None = None,
) -> TracerState:
    """Initialize the active tracers with zero fields; the inactive ones stay ``None``."""
    if tracer_config is None:
        tracer_config = TracerConfig.none()
    return TracerState(
        **{
            name: data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
            )
            for name in tracer_config.active_names
        }
    )
