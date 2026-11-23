# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.dycore import iau


WINDOW = 10800.0  # 3 h, a typical dt_iau
DTIME = 10.0


def test_weights_over_the_window_add_up_to_one():
    """The top-hat window integrates to 1, so the midpoint weights of all steps in it do too."""
    steps = int(WINDOW / DTIME)
    total = sum(
        iau.compute_iau_weight((step + 0.5) * DTIME, DTIME, WINDOW).dynamics
        for step in range(steps)
    )
    assert total == pytest.approx(1.0, rel=1e-6)


def test_top_hat_weight_is_the_step_fraction_of_the_window():
    """In ICON's operation order, (1 / dt_iau) * dt (mo_iau.f90:377, 399-413), which can differ
    from dt / dt_iau in the last bit."""
    weight = iau.compute_iau_weight(1234.5, DTIME, WINDOW)
    assert weight.is_active
    assert weight.dynamics == (1.0 / WINDOW) * DTIME


def test_window_is_closed_at_its_end_and_off_after_it():
    at_end = iau.compute_iau_weight(WINDOW, DTIME, WINDOW)
    after = iau.compute_iau_weight(WINDOW + 1e-9, DTIME, WINDOW)
    assert at_end.is_active
    assert after == iau.IauWeight(is_active=False, dynamics=0.0)


def test_substeps_are_evaluated_at_their_midpoints():
    """ICON's substep loop: time = end of step + (nstep - n - 0.5) * dt, nstep = 1..n. With the
    step ending 25 s after the window, the midpoints are window - 20, -10, +0, +10, +20 s: the
    third substep sits exactly on the (closed) end. Evaluated at substep ends it would be off."""
    weights = iau.compute_iau_weights_for_substeps(
        time_at_end_of_step=WINDOW + 25.0,
        dtime=DTIME,
        ndyn_substeps=5,
        window=WINDOW,
    )
    assert [w.is_active for w in weights] == [True, True, True, False, False]
    assert weights[0] == iau.compute_iau_weight(WINDOW - 20.0, DTIME, WINDOW)


def test_window_can_end_inside_a_time_step():
    """The window may close between substeps: the later substeps of that step get no increment."""
    weights = iau.compute_iau_weights_for_substeps(
        time_at_end_of_step=WINDOW + 20.0,
        dtime=DTIME,
        ndyn_substeps=5,
        window=WINDOW,
    )
    assert [w.is_active for w in weights] == [True, True, True, False, False]


def test_only_the_top_hat_window_is_accepted():
    """ICON's type_iau_wgt: 1 top-hat, 2 sin², 3 sin (mo_iau.f90:361-374); only 1 is ported."""
    iau.check_window_function(iau.TOP_HAT_WINDOW)
    for type_iau_wgt in (2, 3):
        with pytest.raises(NotImplementedError, match="type_iau_wgt"):
            iau.check_window_function(type_iau_wgt)
