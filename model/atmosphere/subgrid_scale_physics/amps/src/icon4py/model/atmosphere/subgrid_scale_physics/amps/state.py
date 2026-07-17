# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-column state bundles mirroring the Fortran qrpv/qipv/qapv property-
vector-bin arrays, per
docs/superpowers/facts/m1/state-packing-si-cgs.md ("F4" below) §1
packing, plus a thermodynamic-state counterpart -- and the numpy <-> gt4py
`(Cell, K)` field conversion used to hand these bundles to DSL code.

Four dataclasses share one shape convention -- `values: np.ndarray` shaped
`(nprops, nbins, ncat, npoints)`:

* `LiquidState`  -- qrpv, props = `LiquidPPV`  (4 members,  F4 SS1.2)
* `IceState`     -- qipv, props = `IcePPV`     (16 members, F4 SS1.3)
* `AerosolState` -- qapv, props = `AerosolPPV` (3 members,  F4 SS1.4)
* `ThermoState`  -- the per-column thermo block computed BEFORE any
  hydrometeor loop (F4 SS1.1, `Z_LOOP_01` lines 1625-1672): ptotv, tv,
  thv, piv, pbv, moist_denv, qvv, thetav, wbv, momv. These are NOT a
  Fortran qXpv property-vector space (F4 SS1.1 has no such array behind
  them -- they are plain per-k scalars), so `ThermoProp` below is a
  LOCAL ordering convention (unlike `LiquidPPV`/`IcePPV`/`AerosolPPV`,
  it is not one of Task 4's `core/index_maps.py` enums), pinned to
  `nbins=1` (there is no bin axis for thermo state) purely so
  `ThermoState` can reuse the exact same `to_fields()`/`from_fields()`
  machinery and field-naming convention as the three PPV bundles below.

Axis 0 (property) order is exactly the Fortran 1-based PPV index (0-based
array index `i` <-> Fortran index `i+1`), sourced from Task 4's
`core/index_maps.py` enums (`IcePPV`/`LiquidPPV`/`AerosolPPV`) via
`FortranIndex.value` -- NOT re-derived or re-numbered here. Axis 2
(category) is only ever exercised at `ncat=1` anywhere in F4 (every
category loop in F4 SS1.2-SS1.4 is annotated "assume(d) to be 1" in the
Fortran source), so `to_fields()`/`from_fields()` require `ncat == 1` and
raise `NotImplementedError` otherwise -- a real `ncat>1` bundle is out of
scope for this task, not silently mishandled.

`.to_fields()`/`.from_fields()` flatten/unflatten axis 3 (`npoints`) into
gt4py `(Cell, K)` fields via a fixed, explicit convention: point index
`p = cell*nlev + level` (row-major / C-order reshape to `(ncells, nlev)`)
-- this convention is this module's OWN choice (F4's Fortran runs a
single column, i.e. only a `k` axis; there is no existing (cell,k)-
flattening precedent to transcribe here), documented since it is
load-bearing for every caller. One gt4py field is produced per
(property, bin) pair, named `f"{GROUP}_{prop_name}_{bin:02d}"` (`GROUP`
in `{"liquid", "ice", "aerosol", "thermo"}`, `prop_name` the enum
member's `.name`, e.g. `"liquid_rmt_q_00"`, `"ice_imas_q_15"`,
`"thermo_ptotv_00"`).

Round-trip (`from_fields(to_fields(nlev), ...)`) is lossless: no
arithmetic happens anywhere in either direction, only reshapes/copies of
float64 data -- so it is exact (`np.array_equal`), never merely close.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import ClassVar, Self

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps
from icon4py.model.common import dimension as dims


class ThermoProp(enum.IntEnum):
    """Per-column thermo-state fields, F4 SS1.1 (`Z_LOOP_01` thermo
    block, `scale_atmos_phy_mp_amps.F90` lines 1625-1672), numbered in
    the order each is computed there (NOT a Task 4 `core/index_maps.py`
    enum -- see module docstring for why)."""

    ptotv = 1  # pressure; line 1641 `ptotv(k) = PRES(k,i,j)`
    tv = 2  # temperature; line 1643 `tv(k) = TEMP(k,i,j)`
    thv = 3  # potential temperature; line 1645
    piv = 4  # exner function; line 1647
    pbv = 5  # unknown/perturbation pressure, always 0; line 1649
    moist_denv = 6  # moist-air density; line 1651
    qvv = 7  # vapor mixing ratio, per moist air; line 1653
    thetav = 8  # virtual potential temperature; line 1657
    wbv = 9  # w at full grid; line 1659
    momv = 10  # w at half grid / MOMZ; line 1661


def _sorted_props(prop_cls: type[enum.IntEnum]) -> tuple[enum.IntEnum, ...]:
    """Fortran 1-based index order (0-based tuple index `i` <-> Fortran
    index `i+1`) for any 1-based IntEnum property-index space."""
    return tuple(sorted(prop_cls, key=lambda member: member.value))


@dataclasses.dataclass(frozen=True)
class _BinnedState:
    """Shared array-shape/round-trip machinery for `LiquidState`/
    `IceState`/`AerosolState`/`ThermoState` -- see module docstring for
    the shape and field-naming convention. Not instantiated directly
    (subclasses set `GROUP`/`PROPS`)."""

    values: np.ndarray  # (nprops, nbins, ncat, npoints), float64

    GROUP: ClassVar[str]
    PROPS: ClassVar[tuple[enum.IntEnum, ...]]

    def __post_init__(self) -> None:
        if type(self) is _BinnedState:
            raise TypeError(
                "_BinnedState is not meant to be instantiated directly; use a subclass."
            )
        if self.values.ndim != 4:
            raise ValueError(
                f"{type(self).__name__}.values must be 4D (nprops, nbins, ncat, npoints); "
                f"got shape {self.values.shape}"
            )
        if self.values.shape[0] != len(self.PROPS):
            raise ValueError(
                f"{type(self).__name__}.values.shape[0] (nprops) must equal "
                f"len({type(self).__name__}.PROPS)={len(self.PROPS)}; got {self.values.shape[0]}"
            )

    @property
    def nprops(self) -> int:
        return self.values.shape[0]

    @property
    def nbins(self) -> int:
        return self.values.shape[1]

    @property
    def ncat(self) -> int:
        return self.values.shape[2]

    @property
    def npoints(self) -> int:
        return self.values.shape[3]

    @classmethod
    def _expected_name(cls, prop: enum.IntEnum, bin_index: int) -> str:
        return f"{cls.GROUP}_{prop.name}_{bin_index:02d}"

    def to_fields(self, nlev: int) -> dict[str, gtx.Field]:
        """One `(Cell, K)` gt4py field per (property, bin) -- see module
        docstring for the naming/flattening convention.

        Args:
            nlev: number of vertical levels; `npoints` must be an exact
                multiple of it (`ncells = npoints // nlev`).
        """
        if self.ncat != 1:
            raise NotImplementedError(
                f"{self.GROUP}: to_fields() only supports ncat=1 (F4: every category loop is "
                f"'assume(d) to be 1'); got ncat={self.ncat}. See module docstring."
            )
        if nlev <= 0:
            raise ValueError(f"nlev must be positive; got {nlev}")
        if self.npoints % nlev != 0:
            raise ValueError(f"npoints={self.npoints} is not divisible by nlev={nlev}")
        ncells = self.npoints // nlev

        fields: dict[str, gtx.Field] = {}
        for prop_idx, prop in enumerate(self.PROPS):
            for bin_index in range(self.nbins):
                flat = self.values[prop_idx, bin_index, 0, :]
                arr2d = np.ascontiguousarray(flat.reshape(ncells, nlev))
                fields[self._expected_name(prop, bin_index)] = gtx.as_field(
                    (dims.CellDim, dims.KDim), arr2d
                )
        return fields

    @classmethod
    def from_fields(cls, fields: Mapping[str, gtx.Field], *, nbins: int) -> Self:
        """Inverse of `to_fields()`. `nbins` cannot be inferred from
        `fields` alone (a group with a partially-populated bin range
        would look ambiguous), so callers supply it explicitly."""
        if nbins <= 0:
            raise ValueError(f"nbins must be positive; got {nbins}")

        missing = [
            cls._expected_name(prop, b)
            for prop in cls.PROPS
            for b in range(nbins)
            if cls._expected_name(prop, b) not in fields
        ]
        if missing:
            raise KeyError(f"{cls.GROUP}: from_fields() missing keys: {missing}")

        sample = next(iter(fields.values())).asnumpy()
        if sample.ndim != 2:
            raise ValueError(f"expected 2D (Cell, K) fields; got shape {sample.shape}")
        ncells, nlev = sample.shape
        npoints = ncells * nlev

        nprops = len(cls.PROPS)
        values = np.empty((nprops, nbins, 1, npoints), dtype=np.float64)
        for prop_idx, prop in enumerate(cls.PROPS):
            for bin_index in range(nbins):
                name = cls._expected_name(prop, bin_index)
                arr2d = fields[name].asnumpy()
                if arr2d.shape != (ncells, nlev):
                    raise ValueError(f"{name}: shape {arr2d.shape} != expected {(ncells, nlev)}")
                values[prop_idx, bin_index, 0, :] = arr2d.reshape(npoints)

        return cls(values=values)


@dataclasses.dataclass(frozen=True)
class LiquidState(_BinnedState):
    """qrpv equivalent, F4 SS1.2. `PROPS` = `LiquidPPV` in Fortran
    1-based index order: `rmt_q`, `rcon_q`, `rmat_q`, `rmas_q`."""

    GROUP: ClassVar[str] = "liquid"
    PROPS: ClassVar[tuple[enum.IntEnum, ...]] = _sorted_props(index_maps.LiquidPPV)


@dataclasses.dataclass(frozen=True)
class IceState(_BinnedState):
    """qipv equivalent, F4 SS1.3. `PROPS` = `IcePPV` in Fortran 1-based
    index order: `imt_q`, `icon_q`, `ivcs_q`, `iacr_q`, `iccr_q`,
    `idcr_q`, `iag_q`, `icg_q`, `inex_q`, `imr_q`, `imc_q`, `imw_q`,
    `imat_q`, `imas_q`, `ima_q`, `imf_q`."""

    GROUP: ClassVar[str] = "ice"
    PROPS: ClassVar[tuple[enum.IntEnum, ...]] = _sorted_props(index_maps.IcePPV)


@dataclasses.dataclass(frozen=True)
class AerosolState(_BinnedState):
    """qapv equivalent, F4 SS1.4. `PROPS` = `AerosolPPV` in Fortran
    1-based index order: `amt_q`, `acon_q`, `ams_q`."""

    GROUP: ClassVar[str] = "aerosol"
    PROPS: ClassVar[tuple[enum.IntEnum, ...]] = _sorted_props(index_maps.AerosolPPV)


@dataclasses.dataclass(frozen=True)
class ThermoState(_BinnedState):
    """F4 SS1.1 per-column thermo block -- see module docstring for why
    this reuses the PPV-bundle machinery with a fixed `nbins=1`
    (`ThermoProp`, not a Task 4 index map)."""

    GROUP: ClassVar[str] = "thermo"
    PROPS: ClassVar[tuple[enum.IntEnum, ...]] = _sorted_props(ThermoProp)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.values.shape[1] != 1:
            raise ValueError(
                f"ThermoState has no bin axis; values.shape[1] (nbins) must be 1, "
                f"got {self.values.shape[1]}"
            )

    @classmethod
    def from_fields(cls, fields: Mapping[str, gtx.Field], *, nbins: int = 1) -> Self:
        if nbins != 1:
            raise ValueError(f"ThermoState.from_fields: nbins must be 1 (no bin axis); got {nbins}")
        return super().from_fields(fields, nbins=1)
