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
import numpy.testing as np_testing
import numpy.typing as npt
import pytest
from typing_extensions import Buffer

from icon4py.model.common import model_options
from icon4py.model.testing import config


logger = logging.getLogger(__file__)


def dallclose(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
    equal_nan: bool = False,
) -> bool:
    """
    'numpy.allclose', but with double precision default tolerances.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def assert_dallclose(
    actual: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
    equal_nan: bool = False,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    """
    'numpy.testing.assert_allclose', but with double precision default tolerances.
    """
    if config.DALLCLOSE_PRINT_INSTEAD_OF_FAIL:
        # Non-blocking version: prints max diff instead of raising errors.
        # Prints red if delta > 0, green otherwise.
        max_diff = np.max(np.abs(np.asarray(actual) - np.asarray(desired)))
        color = "\033[1;31m" if max_diff > 0 else "\033[32m"
        print(f"{color}{err_msg} max diff {max_diff}\033[0m")
    else:
        np_testing.assert_allclose(
            actual,  # type: ignore[arg-type]
            desired,  # type: ignore[arg-type]
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
        )


def is_sorted(array: np.ndarray) -> bool:
    return bool((array[:-1] <= array[1:]).all())


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]  # type: ignore[arg-type]


def get_fixture_value(name: str, item: pytest.Item) -> Any:
    if name in item.fixturenames:  # type: ignore[attr-defined]
        # Get the fixture value using the item's fixture manager
        return item._request.getfixturevalue(name)  # type: ignore[attr-defined]

    return None


def get_backend_fixture_value(item: pytest.Item) -> gtx_typing.Backend | None:
    backend = get_fixture_value("backend", item)
    if backend is not None:
        return backend
    backend_like = get_fixture_value("backend_like", item)
    if backend_like is not None:
        return model_options.customize_backend(None, backend_like)
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
