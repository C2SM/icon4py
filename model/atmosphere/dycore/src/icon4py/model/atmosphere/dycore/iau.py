# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Weights of the incremental analysis update (IAU) for the dynamical core.

Port of `compute_iau_wgt` and its window functions from ICON's `src/atm_dyn_iconam/mo_iau.f90`
(lines 332-389 and 399-460). The weight accumulated for tracer advection (`iau_wgt_adv`) is not
ported: it belongs to the caller that runs advection.
"""

from __future__ import annotations

import dataclasses
import enum
import math

from icon4py.model.common import type_alias as ta


class IauWindowFunction(enum.IntEnum):
    """Shape of the IAU weight over the IAU window; ICON's `type_iau_wgt` (mo_initicon_nml.f90)."""

    TOP_HAT = 1
    SIN2 = 2
    SIN = 3


@dataclasses.dataclass(frozen=True)
class IauWeight:
    is_active: bool
    """Whether the time lies in the IAU window (ICON's `is_iau_active`)."""
    dynamics: ta.wpfloat
    """Weight of the increments in one dynamics step (ICON's `iau_wgt_dyn`)."""


def _window_function(
    window_function: IauWindowFunction, window: ta.wpfloat, time: ta.wpfloat
) -> ta.wpfloat:
    """`iau_top_hat`, `iau_sin2` and `iau_sin` (mo_iau.f90:399-460); each integrates to 1 over the window."""
    if time > window:
        return 0.0
    match window_function:
        case IauWindowFunction.TOP_HAT:
            return 1.0 / window
        case IauWindowFunction.SIN2:
            return (2.0 / window) * math.sin(math.pi * time / window) ** 2
        case IauWindowFunction.SIN:
            return ((math.pi / 2.0) / window) * math.sin(math.pi * time / window)
    raise ValueError(f"Invalid IAU window function {window_function!r}: must be 1, 2 or 3.")


def compute_iau_weight(
    time: ta.wpfloat,
    dtime: ta.wpfloat,
    window: ta.wpfloat,
    window_function: IauWindowFunction,
) -> IauWeight:
    """
    IAU weight for a dynamics step of length `dtime` at `time` seconds into the IAU window
    (`compute_iau_wgt`, mo_iau.f90:332-389).

    The window is active while `time <= window`, as in ICON; outside it the weight is 0.
    """
    if time > window:
        return IauWeight(is_active=False, dynamics=0.0)
    return IauWeight(
        is_active=True, dynamics=_window_function(window_function, window, time) * dtime
    )


def compute_iau_weights_for_substeps(
    time_at_end_of_step: ta.wpfloat,
    dtime: ta.wpfloat,
    ndyn_substeps: int,
    window: ta.wpfloat,
    window_function: IauWindowFunction,
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
            time_at_end_of_step + (substep - ndyn_substeps - 0.5) * dtime,
            dtime,
            window,
            window_function,
        )
        for substep in range(1, ndyn_substeps + 1)
    ]
