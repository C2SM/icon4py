# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from icon4py.model.common.model_options import customize_backend
import pytest
from icon4py.model.common.model_backends import BACKENDS, make_custom_dace_backend


@pytest.mark.parametrize(
    "backend_kind, device", [("gtfn", "cpu"), ("gtfn", "gpu"), ("dace", "cpu"), ("dace", "gpu")]
)
def test_custom_backend(backend_kind, device):
    backend_options = {
        "backend_kind": backend_kind,
        "device": device,
    }
    backend = customize_backend(**backend_options)
    default_backend = backend_kind + "_" + device
    assert str(BACKENDS[default_backend]) == str(backend)
