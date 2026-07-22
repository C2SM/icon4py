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
    disables a process via `dt_xxx == 0`, we use an explicit `enable_process`
    flag.
      - `enable_process`: explicit on/off switch for the process.
      - `interval`   (`dt_xxx`): firing interval; must be > 0 when enabled.
      - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
        `[start, end)` window during which the process exists at all.
    """

    interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    enable_process: bool = True

    def is_in_window(self, simulation_current_datetime: datetime.datetime) -> bool:
        return self.start_date <= simulation_current_datetime < self.end_date

    def is_active(self, simulation_current_datetime: datetime.datetime) -> bool:
        """True if the process's mtime-event fires on the given step.

        Equivalent to AES `isCurrentEventActive(ev_xxx, datetime)`. Fires only
        when the elapsed time is an exact integer multiple of the interval.
        """
        if (
            not self.enable_process
            or self.interval <= datetime.timedelta(0)
            or simulation_current_datetime < self.start_date
        ):
            return False
        elapsed = simulation_current_datetime - self.start_date
        return elapsed % self.interval == datetime.timedelta(0)
