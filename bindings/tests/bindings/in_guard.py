# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard that real-granule tests do not write arguments declared IN."""

import typing

import numpy as np
import pytest

from icon4py.bindings import diffusion_wrapper, dycore_wrapper, icon4py_export
from icon4py.tools import py2fgen


def in_field_names(fun) -> set[str]:
    """Float field arguments of a binding that are declared IN."""
    hints = typing.get_type_hints(fun.__wrapped__, include_extras=True)
    names = set()
    for name, descriptor in fun.param_descriptors.items():
        if not isinstance(descriptor, py2fgen.ArrayParamDescriptor) or descriptor.dtype not in (
            py2fgen.FLOAT32,
            py2fgen.FLOAT64,
        ):
            continue
        base, boundary = icon4py_export._split_boundary(hints[name])
        if (
            icon4py_export._get_gt4py_type(base) is not None
            and boundary.intent is icon4py_export.Intent.IN
        ):
            names.add(name)
    return names


def _host_copy(array):
    return array.get() if hasattr(array, "get") else np.array(array, copy=True)


def guard(monkeypatch, module, fun_name: str, guarded_calls: list) -> None:
    """Make `module.fun_name` raise if a call changes ICON's buffer of an argument declared IN."""
    fun = getattr(module, fun_name)
    names = in_field_names(fun)

    def guarded(*, ffi, **kwargs):
        before = {}
        for name in names:
            if (
                kwargs.get(name) is not None
                and (array := py2fgen.as_array(ffi, kwargs[name])) is not None
            ):
                before[name] = (array, _host_copy(array))
        result = fun(ffi=ffi, **kwargs)
        changed = sorted(
            name
            for name, (array, old) in before.items()
            if not np.array_equal(_host_copy(array), old, equal_nan=True)
        )
        guarded_calls.append((fun_name, sorted(before)))
        assert not changed, f"'{fun_name}' changed arguments declared IN: {', '.join(changed)}."
        return result

    monkeypatch.setattr(module, fun_name, guarded)


@pytest.fixture
def in_arguments_unchanged(monkeypatch):
    """
    Fail a real-granule test whose granule writes an argument declared IN.

    Such a write is lost whenever the argument is a converted copy, and ICON's verify mode cannot
    see it. Only meaningful where the arguments are views of ICON's memory (both sides double).
    """
    guarded_calls: list = []
    guard(monkeypatch, dycore_wrapper, "solve_nh_run", guarded_calls)
    guard(monkeypatch, diffusion_wrapper, "diffusion_run", guarded_calls)
    yield guarded_calls
    assert guarded_calls, "No 'solve_nh_run' or 'diffusion_run' call was guarded."
