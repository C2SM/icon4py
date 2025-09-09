# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from icon4py.model.common.model_options import customize_backend
import pytest
from icon4py.model.common.model_backends import (
    BACKENDS,
    make_custom_dace_backend,
    make_custom_gtfn_backend,
)
import gt4py.next as gtx


@pytest.mark.parametrize("backend_factory", [make_custom_gtfn_backend, make_custom_dace_backend])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_custom_backend_backend_options(backend_factory, device):
    backend_options = {
        "backend_factory": backend_factory,
        "device": device,
    }
    backend = customize_backend(backend_options)
    default_backend = str(backend_factory).split("_")[2] + "_" + device
    assert str(BACKENDS[default_backend]) == str(backend)


def test_custom_backend_device():
    device = gtx.DeviceType.CPU
    backend = customize_backend(device)
    default_backend = "gtfn_cpu"
    assert str(BACKENDS[default_backend]) == str(backend)
