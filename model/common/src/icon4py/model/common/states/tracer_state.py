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

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.states.data import COMMON_TRACER_CF_ATTRIBUTES


#: Tracer names sorted by their Fortran index (QV=0, QC=1, QI=2, QR=3, QS=4, QG=5).
_TRACER_FIELDS: tuple[str, ...] = tuple(
    sorted(
        COMMON_TRACER_CF_ATTRIBUTES,
        key=lambda k: COMMON_TRACER_CF_ATTRIBUTES[k].get("icon_var_list_index", 0),  # type: ignore[arg-type]
    )
)


@dataclasses.dataclass(frozen=True)
class TracerConfig:
    """Which tracers are active in the model configuration.

    Each boolean field indicates whether the corresponding tracer is active.
    Used instead of a raw ``ntracer: int`` to provide type-safe tracer selection.
    """

    qv: bool = False
    qc: bool = False
    qi: bool = False
    qr: bool = False
    qs: bool = False
    qg: bool = False

    @classmethod
    def all(cls) -> TracerConfig:
        return cls(qv=True, qc=True, qi=True, qr=True, qs=True, qg=True)

    @classmethod
    def none(cls) -> TracerConfig:
        return cls()

    @classmethod
    def from_ntracer(cls, ntracer: int) -> TracerConfig:
        """Build a ``TracerConfig`` from a Fortran ``ntracer`` count.

        Fortran ICON uses a fixed tracer ordering: QV=0, QC=1, QI=2, QR=3, QS=4, QG=5.
        The first *ntracer* entries in this order are considered active.
        """
        return cls(**{name: i < ntracer for i, name in enumerate(_TRACER_FIELDS)})

    @property
    def nactive(self) -> int:
        return sum(dataclasses.asdict(self).values())

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(name for name in _TRACER_FIELDS if getattr(self, name))

    def __iter__(self) -> Iterator[str]:
        return iter(self.active_names)

    def __len__(self) -> int:
        return self.nactive

    def __contains__(self, name: str) -> bool:
        return name in _TRACER_FIELDS and getattr(self, name)

    def __bool__(self) -> bool:
        return self.nactive > 0


@dataclasses.dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog
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

    def active_fields(self) -> Iterator[tuple[str, fa.CellKField[ta.wpfloat]]]:
        """Yield ``(name, field)`` for each non-``None`` tracer field."""
        for name in _TRACER_FIELDS:
            field = getattr(self, name)
            if field is not None:
                yield name, field

    def active_pairs(
        self, new: TracerState
    ) -> Iterator[tuple[str, fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]]:
        """Yield ``(name, now_field, new_field)`` for each non-``None`` tracer in both states."""
        for name in _TRACER_FIELDS:
            self_field = getattr(self, name)
            new_field = getattr(new, name)
            if self_field is not None and new_field is not None:
                yield name, self_field, new_field
