# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
import sys
from collections.abc import Buffer
from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np
import numpy.testing as np_testing
import numpy.typing as npt
import pytest

from icon4py.model.common import model_backends, model_options
from icon4py.model.common.constants import DP_EPS, VP_EPS
from icon4py.model.common.type_alias import precision, vpfloat
from icon4py.model.testing import config


wp_is_dp = precision == "double"

if wp_is_dp:

    def scale_tol(x):
        """identity for double-precision"""
        return x
else:
    _scale_const = np.log2(VP_EPS) / np.log2(DP_EPS)

    def scale_tol(x):
        """scale relative factors according to the reduced range

        Maps 1->1, \\epsilon_d->\\epsilon_s"""
        return np.exp(_scale_const * np.log(x))


def _max_diffs(actual: np.ndarray, desired: np.ndarray) -> tuple[float, float]:
    """
    Max absolute and max relative difference, for choosing 'atol' and 'rtol'.

    The relative difference is 'abs(actual - desired) / abs(desired)', matching
    the 'rtol' term of 'numpy.allclose'. It is 'inf' where 'desired' is zero and
    'actual' is not, since no 'rtol' can cover that case, only 'atol'.
    """
    if actual.size == 0 or desired.size == 0:
        return 0.0, 0.0
    abs_diff = np.abs(actual.astype(np.float64) - desired.astype(np.float64))
    denominator = np.abs(desired.astype(np.float64))
    rel_diff = np.divide(
        abs_diff,
        denominator,
        out=np.full(abs_diff.shape, np.inf),
        where=denominator != 0.0,
    )
    rel_diff[abs_diff == 0.0] = 0.0
    return float(np.max(abs_diff)), float(np.max(rel_diff))


def get_mpi_comparison_tolerance(
    backend: gtx_typing.Backend | None,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Return (atol, rtol) for single-rank vs multi-rank field comparisons.

    When ``ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE`` is set and the backend is a
    known-good CPU backend (``gtfn`` or ``dace``), tolerances are overridden to
    zero for bitwise-exact comparison. Otherwise the caller-supplied tolerances
    are returned unchanged.
    """
    if os.environ.get("ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE") == "1" and (
        model_backends.is_cpu_backend(backend) and (is_gtfn_backend(backend) or is_dace(backend))
    ):
        return 0.0, 0.0
    return atol, rtol


def dallclose(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    rtol: vpfloat = scale_tol(5e3) * VP_EPS,  # for double ≈ 1.11e-12
    atol: vpfloat = 0.0,
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
    rtol: vpfloat = scale_tol(5e3) * VP_EPS,  # for double ≈ 1.11e-12
    atol: vpfloat = 0.0,
    equal_nan: bool = False,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    """
    'numpy.testing.assert_allclose', but with double precision default tolerances.
    """
    actual_arr = np.asarray(actual)
    desired_arr = np.asarray(desired)
    if config.DALLCLOSE_PRINT_INSTEAD_OF_FAIL:
        # Non-blocking version: prints max diffs instead of raising errors.
        # Prints red if delta > 0, green otherwise.
        # Goes to stderr: pytest-xdist discards worker stdout when capturing is
        # off ('-s'), which is how the nox sessions run pytest.
        max_abs_diff, max_rel_diff = _max_diffs(actual_arr, desired_arr)
        color = "\033[1;31m" if max_abs_diff > 0 else "\033[32m"
        # Prefix with the test id: under pytest-xdist these prints interleave
        # with the terminal reporter, so they cannot be attributed to a test
        # from their position in the output alone.
        test_id = os.environ.get("PYTEST_CURRENT_TEST", "").partition(" (")[0]
        print(
            f"{color}{test_id} {err_msg} max abs diff {max_abs_diff} max rel diff {max_rel_diff}\033[0m",
            file=sys.stderr,
        )
    else:
        np_testing.assert_allclose(
            actual_arr,
            desired_arr,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
        )


def is_sorted(array: np.ndarray) -> bool:
    return bool((array[:-1] <= array[1:]).all())


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]


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
