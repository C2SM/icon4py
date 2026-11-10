# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

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
