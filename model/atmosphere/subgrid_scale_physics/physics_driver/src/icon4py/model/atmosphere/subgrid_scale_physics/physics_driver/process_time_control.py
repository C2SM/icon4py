# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-process time control for the physics driver."""

from __future__ import annotations

import dataclasses
import datetime


@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """icon4py analogue of the per-process time fields in AES `aes_phy_tc`.

    Mirrors `mo_aes_phy_main.f90` semantics, with one deviation: where AES
    disables a process via `dt_xxx == 0`, icon4py leaves a disabled process out
    of the driver's process list entirely, so it is never constructed and its
    stencils are never compiled.
      - `interval`   (`dt_xxx`): firing interval; must be > 0.
      - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
        `[start, end)` window during which the process exists at all.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def is_in_window(self, simulation_current_datetime: datetime.datetime) -> bool:
        return self.start_date <= simulation_current_datetime < self.end_date

    def is_active(self, simulation_current_datetime: datetime.datetime) -> bool:
        """True if the process's mtime-event fires on the given step.

        Equivalent to AES `isCurrentEventActive(ev_xxx, datetime)`. Fires only
        when the elapsed time is an exact integer multiple of the interval.
        """
        if self.interval <= datetime.timedelta(0) or simulation_current_datetime < self.start_date:
            return False
        elapsed = simulation_current_datetime - self.start_date
        return elapsed % self.interval == datetime.timedelta(0)

    def validate_interval(self, dtime: datetime.timedelta) -> None:
        """Fail loud if the firing interval cannot align with the model timestep.

        ``is_active`` fires only when the elapsed time is an exact multiple of
        ``interval``. With a discrete timestep that is observed only when
        ``interval`` is a positive integer multiple of ``dtime`` -- otherwise the
        process fires only at common multiples of both (or never), silently.
        """
        if self.interval <= datetime.timedelta(0):
            raise ValueError(f"time-control interval must be positive, got {self.interval}")
        if self.interval % dtime != datetime.timedelta(0):
            raise ValueError(
                f"time-control interval {self.interval} is not an integer multiple of "
                f"the model timestep {dtime}: the process would fire only at common "
                "multiples of both (or never)"
            )
