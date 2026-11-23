# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Weights of the incremental analysis update (IAU) for the dynamical core, top-hat window only.

Port of `compute_iau_wgt` with `iau_top_hat` from ICON's `src/atm_dyn_iconam/mo_iau.f90`
(lines 332-389 and 399-413). ICON's other windows, sin² and sin (`type_iau_wgt` 2 and 3), are
not ported: MeteoSwiss runs the top-hat window, and callers reject the others. The weight
accumulated for tracer advection (`iau_wgt_adv`) is not ported: it belongs to the caller that
runs advection.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from icon4py.model.common import type_alias as ta


TOP_HAT_WINDOW: Final = 1
"""ICON's `type_iau_wgt` (mo_initicon_nml.f90) of the top-hat window, the only one supported."""


def check_window_function(type_iau_wgt: int) -> None:
    """Raise unless ICON's `type_iau_wgt` is the top-hat window, the only one ported here."""
    if type_iau_wgt != TOP_HAT_WINDOW:
        raise NotImplementedError(
            f"IAU window type_iau_wgt = {type_iau_wgt} is not supported, only the top-hat window"
            f" ({TOP_HAT_WINDOW}); sin² (2) and sin (3) are not ported."
        )


@dataclasses.dataclass(frozen=True)
class IauWeight:
    is_active: bool
    """Whether the time lies in the IAU window (ICON's `is_iau_active`)."""
    dynamics: ta.wpfloat
    """Weight of the increments in one dynamics step (ICON's `iau_wgt_dyn`)."""


def compute_iau_weight(time: ta.wpfloat, dtime: ta.wpfloat, window: ta.wpfloat) -> IauWeight:
    """
    Top-hat IAU weight for a dynamics step of length `dtime` at `time` seconds into the IAU
    window (`compute_iau_wgt` with `iau_top_hat`, mo_iau.f90:332-389, 399-413).

    The window is active while `time <= window`, as in ICON; outside it the weight is 0. Inside
    it the weight is `(1 / window) * dtime`, in ICON's operation order.
    """
    if time > window:
        return IauWeight(is_active=False, dynamics=0.0)
    return IauWeight(is_active=True, dynamics=(1.0 / window) * dtime)


def compute_iau_weights_for_substeps(
    time_at_end_of_step: ta.wpfloat,
    dtime: ta.wpfloat,
    ndyn_substeps: int,
    window: ta.wpfloat,
) -> list[IauWeight]:
    """
    IAU weights of the `ndyn_substeps` dynamics substeps of one time step, in order.

    `time_at_end_of_step` is the IAU time at the end of the time step (ICON: time since the
    experiment start minus the time shift), `dtime` the substep length. Each substep is
    evaluated at its midpoint, as ICON's substep loop does (mo_nh_stepping.f90:3202-3204,
    `(nstep - ndyn_substeps - 0.5) * dt_dyn` for nstep = 1..ndyn_substeps). In ICON's first
    IAU iteration (`iau_iter == 1`) the window is half as long (mo_nh_stepping.f90:3206): pass
    `window / 2` then.
    """
    return [
        compute_iau_weight(
            time_at_end_of_step + (substep - ndyn_substeps - 0.5) * dtime, dtime, window
        )
        for substep in range(1, ndyn_substeps + 1)
    ]
