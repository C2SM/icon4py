# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


def test_package_imports_and_has_version():
    from icon4py.model.atmosphere.subgrid_scale_physics import amps  # noqa: PLC0415

    assert amps.__version__ == "0.2.0"
