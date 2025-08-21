# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from typing import Any

import numpy as np
import pytest
from gt4py.next import backend as gtx_backend
from typing_extensions import Buffer


def dallclose(
    a: np.ndarray, b: np.ndarray, rtol: float = 1.0e-12, atol: float = 0.0, equal_nan: bool = False
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]  # type: ignore[arg-type]


def get_fixture_value(name: str, item: pytest.Item) -> Any:
    if name in item.fixturenames:  # type: ignore[attr-defined]
        # Get the fixture value using the item's fixture manager
        return item._request.getfixturevalue(name)  # type: ignore[attr-defined]

    return None


def is_embedded(backend: gtx_backend.Backend | None) -> bool:
    return backend is None


def is_roundtrip(backend: gtx_backend.Backend | None) -> bool:
    return backend.name == "roundtrip" if backend else False


def is_python(backend: gtx_backend.Backend | None) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_dace(backend: gtx_backend.Backend | None) -> bool:
    return backend.name.startswith("run_dace_") if backend else False


def is_gtfn_backend(backend: gtx_backend.Backend | None) -> bool:
    return "gtfn" in backend.name if backend else False


def is_scalar(param) -> bool:
    scalar_types = (int, float, str, bool)
    return isinstance(param, scalar_types)
