# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Any

import numpy as np
import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _export


def test_from_annotated():
    testee = Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)]

    result = _export._from_annotated(testee)

    assert isinstance(result, py2fgen.ScalarParamDescriptor)
    assert result.dtype == py2fgen.INT32


@pytest.mark.parametrize(
    "testee, expected",
    [
        (float, py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)),
        (np.float32, py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)),
        (int, ValueError),  # no descriptor deducible
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32), str],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
    ],
)
def test_get_param_descriptor_from_annotation(testee, expected):
    def float_param_descriptor_hook(annotation: Any):
        if annotation in (float, np.float32):
            return py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)
        return None

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _export.param_descriptor_from_annotation(
                testee, annotation_descriptor_hook=float_param_descriptor_hook
            )
    else:
        result = _export.param_descriptor_from_annotation(
            testee, annotation_descriptor_hook=float_param_descriptor_hook
        )
        assert result == expected


_INT32_SCALAR = py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)


def _recording_mapping_hook(events: list):
    """A mapping hook whose mappers double their input and carry a `writeback`."""

    def hook(_: Any, __: py2fgen.ParamDescriptor):
        def mapper(value, *, ffi):
            events.append(("map", value))
            return 2 * value

        def writeback(value, mapped, *, ffi):
            events.append(("writeback", value, mapped))

        mapper.writeback = writeback
        return mapper

    return hook


def test_writeback_runs_after_the_function_with_raw_and_mapped_values():
    events = []

    @py2fgen.export(
        param_descriptors={"a": _INT32_SCALAR},
        annotation_mapping_hook=_recording_mapping_hook(events),
    )
    def fun(a: int):
        events.append(("call", a))

    fun(ffi=None, perf_counters=None, a=3)

    assert events == [("map", 3), ("call", 6), ("writeback", 3, 6)]


def test_writeback_does_not_run_when_the_function_raises():
    events = []

    @py2fgen.export(
        param_descriptors={"a": _INT32_SCALAR},
        annotation_mapping_hook=_recording_mapping_hook(events),
    )
    def fun(a: int):
        raise RuntimeError("granule failed")

    with pytest.raises(RuntimeError, match="granule failed"):
        fun(ffi=None, perf_counters=None, a=3)

    assert [e[0] for e in events] == ["map"]


def test_conversion_error_names_the_argument():
    def failing_hook(_: Any, __: py2fgen.ParamDescriptor):
        def mapper(value, *, ffi):
            raise ValueError("cannot convert")

        return mapper

    @py2fgen.export(param_descriptors={"a": _INT32_SCALAR}, annotation_mapping_hook=failing_hook)
    def fun(a: int):
        pass

    with pytest.raises(ValueError, match="cannot convert") as excinfo:
        fun(ffi=None, perf_counters=None, a=3)

    assert "while converting argument 'a' of 'fun'" in excinfo.value.__notes__


def test_writeback_error_names_the_argument():
    def hook(_: Any, __: py2fgen.ParamDescriptor):
        def mapper(value, *, ffi):
            return value

        def writeback(value, mapped, *, ffi):
            raise ValueError("cannot write back")

        mapper.writeback = writeback
        return mapper

    @py2fgen.export(param_descriptors={"a": _INT32_SCALAR}, annotation_mapping_hook=hook)
    def fun(a: int):
        pass

    with pytest.raises(ValueError, match="cannot write back") as excinfo:
        fun(ffi=None, perf_counters=None, a=3)

    assert "while writing back argument 'a' of 'fun'" in excinfo.value.__notes__
