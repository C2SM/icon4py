# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from icon4py.model.common.settings import xp


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "fortran_samples"


def compare_values_shallow(value1, value2):
    # Compare if both are objects with attributes (__dict__)
    if hasattr(value1, "__dict__") and hasattr(value2, "__dict__"):
        return value1.__class__ == value2.__class__  # Only compare class types

    if isinstance(value1, xp.ndarray) and isinstance(value2, xp.ndarray):
        return xp.testing.assert_equal(value1, value2)  # Compare arrays for equality

    if isinstance(value1, dict) and isinstance(value2, dict):
        return value1.keys() == value2.keys()  # Only compare keys

    if isinstance(value1, tuple) and isinstance(value2, tuple):
        return len(value1) == len(value2)  # Only compare length

    # Compare directly for other types
    return value1 == value2


def compare_dicts_shallow(dict1, dict2):
    # Check if both dictionaries have the same keys
    return dict1.keys() == dict2.keys()


def compare_tuples_shallow(tuple1, tuple2):
    # Check if both tuples have the same length
    return len(tuple1) == len(tuple2)


def compare_objects(obj1, obj2):
    # Check if both objects are instances of the same class
    if obj1.__class__ != obj2.__class__:
        return False

    # Shallowly compare the attributes of both objects
    for attr, value in vars(obj1).items():
        other_value = getattr(obj2, attr, None)
        if not compare_values_shallow(value, other_value):
            return False

    return True
