# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import difflib
import os
import pathlib
import subprocess
import sys

import pytest

from icon4py.bindings import all_bindings
from icon4py.tools.py2fgen import _utils


logger = _utils.setup_logger(__name__)

# The .c source is not snapshotted: it is bulky CFFI-generated code that
# varies between cffi versions.
_SNAPSHOT_SUFFIXES = (".py", ".f90", ".h")


# ICON4PY_BINDINGS_ICON_PRECISION -> subdirectory of its snapshot
_ICON_PRECISION_SUBDIRS = {"double": "", "mixed": "mixed"}


def _reference(suffix: str, icon_precision: str) -> pathlib.Path:
    base = pathlib.Path(__file__).parent.resolve() / "references"
    return base / _ICON_PRECISION_SUBDIRS[icon_precision] / f"{all_bindings.LIBRARY_NAME}{suffix}"


def _actual(suffix: str, icon_precision: str) -> pathlib.Path:
    base = pathlib.Path(__file__).parent.resolve() / "references_new"
    return base / _ICON_PRECISION_SUBDIRS[icon_precision] / f"{all_bindings.LIBRARY_NAME}{suffix}"


def diff(reference: pathlib.Path, actual: pathlib.Path) -> bool:
    with pathlib.Path.open(reference) as f:
        reference_lines = f.readlines()
    with pathlib.Path.open(actual) as f:
        actual_lines = f.readlines()

    clean = True
    for line in difflib.context_diff(reference_lines, actual_lines):
        logger.info(f"result line: {line}")
        clean = False
    return clean


@pytest.mark.parametrize("icon_precision", _ICON_PRECISION_SUBDIRS)
def test_references(icon_precision):
    cli_args = []
    for suffix in _SNAPSHOT_SUFFIXES:
        actual = _actual(suffix, icon_precision)
        actual.parent.mkdir(parents=True, exist_ok=True)
        cli_args += [f"--output-{suffix[1:]}", str(actual)]
    # a fresh interpreter, since the bindings read the precision at import
    result = subprocess.run(
        [sys.executable, "-m", "icon4py.bindings.all_bindings", *cli_args],
        env={**os.environ, "ICON4PY_BINDINGS_ICON_PRECISION": icon_precision},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for suffix in _SNAPSHOT_SUFFIXES:
        assert diff(_reference(suffix, icon_precision), _actual(suffix, icon_precision))
