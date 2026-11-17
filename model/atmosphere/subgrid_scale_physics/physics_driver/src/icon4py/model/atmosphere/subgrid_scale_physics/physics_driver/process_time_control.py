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

from icon4py.model.common import time


@dataclasses.dataclass(frozen=True)
class ProcessTimeControl:
    """
    ICON4Py analogue of the per-process time fields in AES `aes_phy_tc`.

    - `driver_dtime`: time step of the driver calling the process
    - `process_dtime`: time step of the process, must be a multiple of `driver_dtime`
    - `window_start`, `window_end`: half-open `[window_start, window_end)` window
      during which the process is active
    - `start_of_simulation`: first step of the driver, only to check `window_start`
    """

    driver_dtime: time.RelativeTime
    process_dtime: time.RelativeTime
    window_start: time.AbsoluteTime
    window_end: time.AbsoluteTime
    start_of_simulation: dataclasses.InitVar[time.AbsoluteTime]

    def __post_init__(self, start_of_simulation: time.AbsoluteTime) -> None:
        if self.process_dtime <= time.RelativeTime(0) or self.process_dtime % self.driver_dtime:
            raise ValueError(
                f"time-control process_dtime {self.process_dtime} must be a positive multiple "
                f"of driver_dtime {self.driver_dtime}"
            )
        if (self.window_start - start_of_simulation) % self.driver_dtime:
            raise ValueError(
                f"time-control window_start {self.window_start} must be a whole number of "
                f"driver_dtime {self.driver_dtime} from the simulation start {start_of_simulation}"
            )

    @property
    def skips_steps(self) -> bool:
        """True when the process does not compute on every driver step."""
        return self.process_dtime > self.driver_dtime

    def is_in_window(self, driver_step_start: time.AbsoluteTime) -> bool:
        """True when the driver's datetime is within the process' window."""
        return self.window_start <= driver_step_start < self.window_end

    def computes_at(self, driver_step_start: time.AbsoluteTime) -> bool:
        """True on multiples of `process_dtime` after `window_start`."""
        if driver_step_start < self.window_start:
            return False
        elapsed = driver_step_start - self.window_start
        return elapsed % self.process_dtime == time.RelativeTime(0)
