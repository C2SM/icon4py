# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import types

import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _runtime, runtime_config


@pytest.mark.parametrize(
    "config, device_enabled, expected",
    [
        (None, False, False),
        (None, True, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_use_device(monkeypatch, config, device_enabled, expected):
    monkeypatch.setattr(runtime_config, "USE_DEVICE", config)
    assert _runtime.use_device(device_enabled) is expected


def test_use_device_disabled_with_device_pointers(monkeypatch):
    monkeypatch.setattr(runtime_config, "USE_DEVICE", False)
    with pytest.raises(RuntimeError, match="PY2FGEN_USE_DEVICE"):
        _runtime.use_device(True)


class _FakeExternalStream:
    def __init__(self, ptr: int):
        self.ptr = ptr


def test_gpu_stream_without_cupy(monkeypatch):
    monkeypatch.setattr(_runtime, "cp", None)
    assert isinstance(_runtime.gpu_stream(42), contextlib.nullcontext)


@pytest.mark.parametrize(
    "external_gpu_stream, expected_ptr",
    [
        (py2fgen.NO_EXTERNAL_GPU_STREAM, None),
        (0, 0),  # the default stream
        (42, 42),
    ],
)
def test_gpu_stream(monkeypatch, external_gpu_stream, expected_ptr):
    fake_cp = types.SimpleNamespace(cuda=types.SimpleNamespace(ExternalStream=_FakeExternalStream))
    monkeypatch.setattr(_runtime, "cp", fake_cp)

    stream = _runtime.gpu_stream(external_gpu_stream)

    if expected_ptr is None:
        assert isinstance(stream, contextlib.nullcontext)
    else:
        assert isinstance(stream, _FakeExternalStream)
        assert stream.ptr == expected_ptr
