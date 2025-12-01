# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from icon4py.model.common.decomposition.definitions import ProcessProperties


def check_comm_size(props: ProcessProperties, sizes: tuple[int, ...] = (1, 2, 4)) -> None:
    if props.comm_size not in sizes:
        pytest.xfail(f"wrong comm size: {props.comm_size}: test only works for comm-sizes: {sizes}")
