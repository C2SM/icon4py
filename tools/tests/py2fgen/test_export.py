# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated

import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _export


def test_from_annotated():
    testee = Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)]

    result = _export._from_annotated(testee)

    assert isinstance(result, py2fgen.ScalarParamDescriptor)
    assert result.dtype == py2fgen.INT32


@pytest.mark.parametrize(
    "testee, expected",
    [
        (float, py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)),
        (int, pytest.raises),  # no descriptor deducible
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32), str],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
    ],
)
def test_get_param_descriptor_from_annotation(testee, expected):
    def param_descriptor_hook(float_param):
        if float_param is float:
            return py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)
        return None

    if expected is pytest.raises:
        with pytest.raises(ValueError):
            _export.param_descriptor_from_annotation(
                testee, annotation_descriptor_hook=param_descriptor_hook
            )
    else:
        result = _export.param_descriptor_from_annotation(
            testee, annotation_descriptor_hook=param_descriptor_hook
        )
        assert result == expected
