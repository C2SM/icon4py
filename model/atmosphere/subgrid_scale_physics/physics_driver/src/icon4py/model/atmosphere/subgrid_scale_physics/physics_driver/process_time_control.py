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
    """ICON4Py analogue of the per-process time fields in AES `aes_phy_tc`.

    - `interval`   (`dt_xxx`): compute interval, > 0
    - `start_date` (`sd_xxx`), `end_date` (`ed_xxx`): half-open
      `[start, end)` window during which the process exists.
    """

    compute_interval: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def __post_init__(self) -> None:
        if self.compute_interval <= datetime.timedelta(0):
            raise ValueError(f"time-control compute_interval must be positive, got {self.compute_interval}")

    def is_in_window(self, step_start: datetime.datetime) -> bool:
        return self.start_date <= step_start < self.end_date

    def computes_at(self, step_start: datetime.datetime) -> bool:
        """True on a whole number of intervals after `start_date` (AES `isCurrentEventActive`)."""
        if step_start < self.start_date:
            return False
        elapsed = step_start - self.start_date
        return elapsed % self.compute_interval == datetime.timedelta(0)
