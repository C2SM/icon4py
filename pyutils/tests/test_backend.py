# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from gt4py.next.iterator import ir as itir

from icon4py.pyutils import backend
from icon4py.pyutils.backend import GTHeader


@pytest.mark.parametrize(
    "input_params, expected_complement",
    [
        ([backend.H_START], [backend.H_END, backend.V_START, backend.V_END]),
        ([backend.H_START, backend.H_END], [backend.V_END, backend.V_START]),
        (backend._DOMAIN_ARGS, []),
        ([], backend._DOMAIN_ARGS),
    ],
)
def test_missing_domain_args(input_params, expected_complement):
    params = [itir.Sym(id=p) for p in input_params]
    domain_boundaries = set(
        map(lambda s: str(s.id), GTHeader._missing_domain_params(params))
    )
    assert len(domain_boundaries) == len(expected_complement)
    assert domain_boundaries == set(expected_complement)
