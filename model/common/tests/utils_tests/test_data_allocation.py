# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common.utils import data_allocation as data_alloc


@pytest.mark.parametrize(
    "backend, value", [(None, False), (gtx.gtfn_cpu, False), (gtx.gtfn_gpu, True)]
)
def test_is_cupy_device(backend, value):
    assert value == data_alloc.is_cupy_device(backend)


@pytest.mark.parametrize(
    "backend, transfer_embedded, device_attr",
    [
        (None, False, True),
        (None, True, True),
        (gtx.gtfn_cpu, True, False),
        (gtx.gtfn_cpu, False, False),
        (gtx.gtfn_gpu, False, True),
        (gtx.gtfn_gpu, True, True),
    ],
)
def test_copy_to_backend_device(backend, transfer_embedded, device_attr):
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        pytest.mark.xfail("Cupy is not installed")
    data = np.array([1, 2, 3])
    transferred_data = data_alloc.copy_to_backend_device(data, backend, transfer_embedded)
    assert hasattr(transferred_data, "device") == device_attr
