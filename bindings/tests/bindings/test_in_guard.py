# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import cffi
import numpy as np
import pytest

from icon4py.bindings import dycore_wrapper
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils

from . import in_guard


def _fake_solve_nh_run(written: str):
    real = dycore_wrapper.solve_nh_run

    def fake(*, ffi, **kwargs):
        py2fgen.as_array(ffi, kwargs[written])[...] = 2.0

    fake.param_descriptors = real.param_descriptors
    fake.__wrapped__ = real.__wrapped__
    return fake


def _call():
    ffi = cffi.FFI()
    kwargs = {
        name: test_utils.array_to_array_info(np.ones((3, 4)), ffi=ffi)
        for name in ("exner_now", "exner_new")
    }
    dycore_wrapper.solve_nh_run(ffi=ffi, perf_counters=None, **kwargs)


def test_guard_reads_the_declared_intents():
    names = in_guard.in_field_names(dycore_wrapper.solve_nh_run)

    assert "exner_now" in names
    assert "exner_new" not in names


def test_guard_rejects_a_write_into_an_in_argument(monkeypatch):
    monkeypatch.setattr(dycore_wrapper, "solve_nh_run", _fake_solve_nh_run("exner_now"))
    in_guard.guard(monkeypatch, dycore_wrapper, "solve_nh_run", [])

    with pytest.raises(AssertionError, match="exner_now"):
        _call()


def test_guard_accepts_a_write_into_an_inout_argument(monkeypatch):
    monkeypatch.setattr(dycore_wrapper, "solve_nh_run", _fake_solve_nh_run("exner_new"))
    guarded_calls = []
    in_guard.guard(monkeypatch, dycore_wrapper, "solve_nh_run", guarded_calls)

    _call()

    assert guarded_calls == [("solve_nh_run", ["exner_now"])]
