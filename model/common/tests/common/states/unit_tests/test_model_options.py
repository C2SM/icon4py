# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from icon4py.model.common import model_backends
from icon4py.model.common.model_options import customize_backend
import pytest

import gt4py.next as gtx


@pytest.mark.parametrize(
    "backend_factory, expected_backend",
    [
        (model_backends.make_custom_gtfn_backend, "gtfn"),
        (model_backends.make_custom_dace_backend, "dace"),
    ],
)
@pytest.mark.parametrize(
    "device, expected_device",
    [(model_backends.CPU, "cpu"), (model_backends.GPU, "gpu")],
)
def test_custom_backend_backend_options(backend_factory, device, expected_backend, expected_device):
    backend_options = {
        "backend_factory": backend_factory,
        "device": device,
    }
    backend = customize_backend(backend_options)
    backend_name = expected_backend + "_" + expected_device
    assert str(model_backends.BACKENDS[backend_name]) == str(backend)


def test_custom_backend_device():
    device = gtx.DeviceType.CPU
    backend = customize_backend(device)
    default_backend = "gtfn_cpu"
    assert str(model_backends.BACKENDS[default_backend]) == str(backend)
