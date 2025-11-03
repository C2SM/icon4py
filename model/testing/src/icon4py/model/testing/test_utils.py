# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import copy
import difflib
import hashlib
import logging
import pathlib
from collections.abc import Sequence
from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import Buffer


logger = logging.getLogger(__file__)


def dallclose(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]  # type: ignore[arg-type]


def get_fixture_value(name: str, item: pytest.Item) -> Any:
    if name in item.fixturenames:  # type: ignore[attr-defined]
        # Get the fixture value using the item's fixture manager
        return item._request.getfixturevalue(name)  # type: ignore[attr-defined]

    return None


def is_embedded(backend: gtx_typing.Backend | None) -> bool:
    return backend is None


def is_roundtrip(backend: gtx_typing.Backend | None) -> bool:
    return backend.name == "roundtrip" if backend else False


def is_python(backend: gtx_typing.Backend | None) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_dace(backend: gtx_typing.Backend | None) -> bool:
    return backend.name.startswith("run_dace_") if backend else False


def is_gtfn_backend(backend: gtx_typing.Backend | None) -> bool:
    return "gtfn" in backend.name if backend else False


def diff(reference: pathlib.Path, actual: pathlib.Path) -> bool:
    with pathlib.Path.open(reference) as f:
        reference_lines = f.readlines()
    with pathlib.Path.open(actual) as f:
        actual_lines = f.readlines()
    result = difflib.context_diff(reference_lines, actual_lines)

    clean = True
    for line in result:
        logger.info(f"result line: {line}")
        clean = False

    return clean


def assert_same_except(properties: Sequence[str], arg1: Any, arg2: Any) -> None:
    assert type(arg1) is type(arg2), f"{arg1} and {arg2} are not of the same type"
    temp = copy.deepcopy(arg2)
    for p in properties:
        assert hasattr(arg1, p), f"object of type {type(arg1)} has not attribute {p} "
        # set these attributes to the same value for comparision later on
        arg1_attr = getattr(arg1, p)
        setattr(temp, p, arg1_attr)
    assert arg1 == temp
