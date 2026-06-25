# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics orchestrator: the ``PhysicsDriver`` and its process / time-control types."""

from __future__ import annotations

import dataclasses
import datetime
import enum
from typing import TYPE_CHECKING, Any

from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import PhysicsStateProtocol


if TYPE_CHECKING:
    from icon4py.model.common.states import prognostic_state, tracer_state


class ForcingMode(enum.IntEnum):
    """Per-process apply switch -- the icon4py analogue of AES ``fc_xxx``.

    Decides whether a process's computed forcing is fed back into the prognostic
    state when the process runs:

    - APPLY:      compute and apply it (``field += tend*dt``); the process affects the run.
    - DIAGNOSTIC: compute it but do NOT apply it -- the outputs stay available for
      inspection/output while the prognostic state is left unchanged ("look, don't touch").

    This composes with ``kind`` tag (``states/model.py``): ForcingMode is
    per-PROCESS, and ``kind`` is whether the field is a tendency or a diagnostic.
    A process is applied only if APPLY mode, and within it only ``kind="tendency"`` fields
    are added to the state.
    """

    DIAGNOSTIC = 0
    APPLY = 1


# TODO (Yilu): this can belong to a standalone file (with the exception of ForcingMode.APPLY)?
@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """icon4py analogue of the per-process fields in AES `aes_phy_tc`.

    Mirrors `mo_aes_phy_main.f90` semantics, with one deviation: where AES
    disables a process via `dt_xxx == 0`, we use an explicit `enable_process`
    flag
      - `enable_process`: explicit on/off switch for the process.
      - `interval`   (`dt_xxx`): firing interval; must be > 0 when enabled.
      - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
        `[start, end)` window during which the process exists at all.
      - `forcing_mode` (`fc_xxx`): DIAGNOSTIC (compute only) or APPLY.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    enable_process: bool = True
    forcing_mode: ForcingMode = ForcingMode.APPLY

    def is_in_window(self, now: datetime.datetime) -> bool:
        return self.start_date <= now < self.end_date

    def is_active(self, now: datetime.datetime) -> bool:
        """True if the process's mtime-event fires on the given step.

        Equivalent to AES `isCurrentEventActive(ev_xxx, datetime)`. Fires only
        when the elapsed time is an exact integer multiple of the interval.
        """
        if (
            not self.enable_process
            or self.interval <= datetime.timedelta(0)
            or now < self.start_date
        ):
            return False
        elapsed = (now - self.start_date).total_seconds()
        interval_s = self.interval.total_seconds()
        quotient = elapsed / interval_s
        return abs(quotient - round(quotient)) == 0.0


@dataclasses.dataclass
class PhysicsProcess:
    """A registered physics process: a component, its state adapter, and its time control.

    The component is the per-process adapter (e.g. ``MuphysComponent``); it
    implements the generic ``Component`` protocol, which is how the driver types it.
    The state adapter is process-specific (it translates the prognostic state to/from
    *this* component's contract), so it is bundled per process rather than shared.
    """

    name: str
    component: Component
    state: PhysicsStateProtocol
    time_control: ProcessTimeControl


class PhysicsDriver:
    """Physics orchestrator. icon4py analogue of `aes_phy_main`."""

    def __init__(
        self,
        processes: list[PhysicsProcess],
    ) -> None:
        self._processes = processes
        self._recycle_cache: dict[str, dict[str, Any]] = {}

    def run(
        self,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_state.TracerState,
        dt: float,
        now: datetime.datetime,
    ) -> None:
        # TODO (Yilu): currently, ForcingMode is not applied, because muphys is always APPLY mode.
        # TODO (Yilu): later on, when a non-APPLY process exits
        for proc in self._processes:
            state = proc.state
            state.gather_from_prognostic(prognostic, tracers)
            tc = proc.time_control
            if not tc.enable_process:
                continue
            in_window = tc.is_in_window(now)
            if in_window and tc.is_active(now):
                # compute: run the component (e.g. muphys) on this process's physics state
                outputs = proc.component(state.as_component_input(), now)
                self._recycle_cache[proc.name] = outputs
            elif in_window:
                # recycle
                outputs = self._recycle_cache[proc.name]
            else:
                # no forcing
                continue
            state.scatter_to_prognostic(prognostic, outputs, dt)
