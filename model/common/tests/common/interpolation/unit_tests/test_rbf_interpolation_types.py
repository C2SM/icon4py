# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.interpolation import rbf_interpolation as rbf


@pytest.mark.level("unit")
def test_default_rbf_kernel_uses_interpolation_kernel_values() -> None:
    assert rbf.DEFAULT_RBF_KERNEL == {
        rbf.RBFDimension.CELL: rbf.InterpolationKernel.GAUSSIAN,
        rbf.RBFDimension.EDGE: rbf.InterpolationKernel.INVERSE_MULTIQUADRATIC,
        rbf.RBFDimension.VERTEX: rbf.InterpolationKernel.GAUSSIAN,
        rbf.RBFDimension.GRADIENT: rbf.InterpolationKernel.GAUSSIAN,
    }
    assert all(
        isinstance(kernel, rbf.InterpolationKernel) for kernel in rbf.DEFAULT_RBF_KERNEL.values()
    )
