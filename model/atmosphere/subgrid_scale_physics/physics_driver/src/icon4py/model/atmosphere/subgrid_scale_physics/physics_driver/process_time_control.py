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
    """
    ICON4Py analogue of the per-process time fields in AES `aes_phy_tc`.

    - `driver_dtime`: time step of the driver calling the process
    - `process_dtime`: time step of the process, must be a multiple of `driver_dtime`
    - `start_date`, `end_date`: half-open `[start, end)` window
      during which the process is active
    """

    driver_dtime: datetime.timedelta
    process_dtime: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def __post_init__(self) -> None:
        for name in ("driver_dtime", "process_dtime"):
            if (value := getattr(self, name)) <= datetime.timedelta(0):
                raise ValueError(f"time-control {name} must be positive, got {value}")
        if self.process_dtime % self.driver_dtime != datetime.timedelta(0):
            raise ValueError(
                f"time-control process_dtime {self.process_dtime} must be a multiple of "
                f"driver_dtime {self.driver_dtime}"
            )

    @property
    def skips_steps(self) -> bool:
        """True when the process does not compute on every driver step."""
        return self.process_dtime > self.driver_dtime

    def is_in_window(self, driver_step_start_datetime: datetime.datetime) -> bool:
        """True when the driver's datetime is within the process' window."""
        return self.start_date <= driver_step_start_datetime < self.end_date

    def computes_at(self, driver_step_start_datetime: datetime.datetime) -> bool:
        """True on a whole number of intervals after `start_date` (AES `isCurrentEventActive`)."""
        if driver_step_start_datetime < self.start_date:
            return False
        elapsed = driver_step_start_datetime - self.start_date
        return elapsed % self.process_dtime == datetime.timedelta(0)
