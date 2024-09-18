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


def compare_objects(obj1: object, obj2: object) -> bool:
    # Check if both objects are instances of the same class
    if obj1.__class__ != obj2.__class__:
        return False

    # Compare the attributes of both objects
    for attr, value in vars(obj1).items():
        other_value = getattr(obj2, attr, None)

        if hasattr(value, "__dict__") and hasattr(other_value, "__dict__"):
            # special case for arrays
            if hasattr(value, "ndarray") and hasattr(other_value, "ndarray"):
                return xp.all(value == other_value)
            return value.__dict__ == other_value.__dict__

        if value != other_value:
            return False

    return True
