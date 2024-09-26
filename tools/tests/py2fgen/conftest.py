# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from gt4py.next.embedded.nd_array_field import NdArrayField
from icon4py.model.common.settings import xp


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "fortran_samples"


def compare_values_shallow(value1, value2, obj_name="value"):
    # Handle comparison of NdArrayField objects
    if isinstance(value1, NdArrayField) and isinstance(value2, NdArrayField):
        try:
            xp.testing.assert_equal(value1.ndarray, value2.ndarray)  # Compare arrays for equality
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Handle comparison of dictionaries
    if isinstance(value1, dict) and isinstance(value2, dict):
        if value1.keys() != value2.keys():
            return False, f"Dict keys mismatch for {obj_name}: {value1.keys()} != {value2.keys()}"
        for key in value1:
            result, error_message = compare_values_shallow(
                value1[key], value2[key], f"{obj_name}[{key}]"
            )
            if not result:
                return False, error_message
        return True, None

    # Handle comparison of tuples
    if isinstance(value1, tuple) and isinstance(value2, tuple):
        if len(value1) != len(value2):
            return False, f"Tuple length mismatch for {obj_name}: {len(value1)} != {len(value2)}"
        for index, (item1, item2) in enumerate(zip(value1, value2, strict=False)):
            result, error_message = compare_values_shallow(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Handle comparison of objects with attributes (__dict__)
    if hasattr(value1, "__dict__") and hasattr(value2, "__dict__"):
        result, error_message = compare_objects(value1, value2, obj_name)
        if not result:
            return False, error_message
        return True, None

    # Check if both values are instances of numpy scalar types
    if isinstance(value1, xp.ScalarType) and isinstance(value2, xp.ScalarType):
        if value1 != value2:
            return False, f"Value mismatch for {obj_name}: {value1} != {value2}"
        return True, None

    # Handle comparison of numpy/cupy array objects
    if isinstance(value1, xp.ndarray) and isinstance(value2, xp.ndarray):
        try:
            xp.testing.assert_equal(value1, value2)  # Compare arrays for equality
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Direct comparison for other types
    if value1 != value2:
        return False, f"Value mismatch for {obj_name}: {value1} != {value2}"

    return True, None


def compare_dicts_shallow(dict1, dict2, obj_name="dict"):
    if dict1.keys() != dict2.keys():
        return False, f"Dict keys mismatch for {obj_name}: {dict1.keys()} != {dict2.keys()}"
    return True, None


def compare_tuples_shallow(tuple1, tuple2, obj_name="tuple"):
    if len(tuple1) != len(tuple2):
        return False, f"Tuple length mismatch for {obj_name}: {len(tuple1)} != {len(tuple2)}"
    return True, None


def compare_objects(obj1, obj2, obj_name="object"):
    # Check if both objects are instances of numpy scalar types
    if isinstance(obj1, xp.ScalarType) and isinstance(obj2, xp.ScalarType):
        if obj1 != obj2:
            return False, f"Value mismatch for {obj_name}: {obj1} != {obj2}"
        return True, None

    # Check if both objects are lists
    if isinstance(obj1, list) and isinstance(obj2, list):
        # Check if lists have the same length
        if len(obj1) != len(obj2):
            return False, f"Length mismatch for {obj_name}: {len(obj1)} != {len(obj2)}"

        # Compare each element in the lists
        for index, (item1, item2) in enumerate(zip(obj1, obj2, strict=False)):
            result, error_message = compare_objects(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Check if both objects are instances of the same class
    if obj1.__class__ != obj2.__class__:
        return False, f"Class mismatch for {obj_name}: {obj1.__class__} != {obj2.__class__}"

    # Shallowly compare the attributes of both objects
    for attr, value in vars(obj1).items():
        other_value = getattr(obj2, attr, None)
        result, error_message = compare_values_shallow(value, other_value, f"{obj_name}.{attr}")
        if not result:
            return False, error_message

    return True, None
