# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""L2 physics orchestrator"""

from __future__ import annotations

import dataclasses
import datetime
import enum
from typing import Any, Protocol

from icon4py.model.common.components.components import Component


class ForcingMode(enum.IntEnum):
    """AES fc_xxx analogue.

    DIAGNOSTIC: compute the forcing but do not apply it to state.
    APPLY:      compute and apply (field += tend*dt).
    """

    DIAGNOSTIC = 0
    APPLY = 1


_ACTIVE_TOLERANCE = 1e-6  # relative tolerance on the modulo, in fractions of `interval`


@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """icon4py analogue of the per-process fields in AES `aes_phy_tc`.

    Mirrors `mo_aes_phy_main.f90` semantics:
      - `interval`   (`dt_xxx`): zero means the process is disabled.
      - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
        `[start, end)` window during which the process exists at all.
      - `forcing_mode` (`fc_xxx`): DIAGNOSTIC (compute only) or APPLY.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    forcing_mode: ForcingMode = ForcingMode.APPLY

    def is_enabled(self) -> bool:
        return self.interval > datetime.timedelta(0)

    def is_in_window(self, now: datetime.datetime) -> bool:
        return self.start_date <= now < self.end_date

    def is_active(self, now: datetime.datetime) -> bool:
        """True if the process's mtime-event fires on the given step.

        Equivalent to AES `isCurrentEventActive(ev_xxx, datetime)`. Uses a
        small relative tolerance to absorb datetime/timedelta floating-point
        jitter accumulated over many steps.

        """
        if now < self.start_date or not self.is_enabled():
            return False
        elapsed = (now - self.start_date).total_seconds()
        interval_s = self.interval.total_seconds()
        quotient = elapsed / interval_s
        return abs(quotient - round(quotient)) < _ACTIVE_TOLERANCE


@dataclasses.dataclass
class PhysicsProcess:
    """A registered physics process: a Component plus its time control."""

    name: str
    component: Component
    time_control: ProcessTimeControl


# TODO(Yilu): this is only a structural protocol. later on this would be a real PhysicsState adapter
# TODO (Yilu)
class PhysicsStateProtocol(Protocol):
    """The slice of PhysicsState that PhysicsDriver depends on."""

    def update_from_prognostic(self, prognostic: Any) -> None: ...
    def as_component_input(self) -> dict[str, Any]: ...
    def scatter_to_prognostic(
        self, prognostic: Any, outputs: dict[str, Any], dt: float
    ) -> None: ...


class PhysicsDriver:
    """L2 physics orchestrator. icon4py analogue of `aes_phy_main`."""

    def __init__(
        self,
        processes: list[PhysicsProcess],
        physics_state: PhysicsStateProtocol,
    ) -> None:
        self._processes = processes
        self._physics_state = physics_state
        self._recycle_cache: dict[str, dict[str, Any]] = {}

    def run(
        self,
        prognostic: Any,
        dt: float,
        now: datetime.datetime,
    ) -> None:
        self._physics_state.update_from_prognostic(prognostic)
        for proc in self._processes:
            tc = proc.time_control
            if not tc.is_enabled():
                continue
            in_window = tc.is_in_window(now)
            if in_window and tc.is_active(now):
                # compute
                outputs = proc.component(self._physics_state.as_component_input(), now)
                self._recycle_cache[proc.name] = outputs
            elif in_window:
                # recycle
                outputs = self._recycle_cache[proc.name]
            else:
                # no forcing
                continue
            self._physics_state.scatter_to_prognostic(prognostic, outputs, dt)
            # Re-sync PhysicsState so the next process sees the post-scatter state
            # (sequential / Lie-Trotter splitting; mirrors AES mo_interface_cloud_mig.f90).
            self._physics_state.update_from_prognostic(prognostic)
