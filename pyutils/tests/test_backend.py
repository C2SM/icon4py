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

from functional.iterator import ir as itir

from icon4py.pyutils import backend
from icon4py.pyutils.backend import GTHeader


def test_missing_domain_args():
    params = [itir.Sym(id=backend.H_START)]
    domain_boundaries = list(
        map(lambda s: s.id, GTHeader._missing_domain_params(params))
    )
    assert len(domain_boundaries) == 3
    assert backend.V_END in domain_boundaries
    assert backend.V_START in domain_boundaries
    assert backend.H_END in domain_boundaries


def test_missing_domain_args_remove_horizontal():
    params = [itir.Sym(id=backend.H_START), itir.Sym(id=backend.H_END)]
    domain_boundaries = list(
        map(lambda s: s.id, GTHeader._missing_domain_params(params))
    )
    assert len(domain_boundaries) == 2
    assert backend.V_END in domain_boundaries
    assert backend.V_START in domain_boundaries


def test_missing_domain_args_is_empty():
    params = [
        itir.Sym(id=backend.V_END),
        itir.Sym(id=backend.V_START),
        itir.Sym(id=backend.H_END),
        itir.Sym(id=backend.H_START),
    ]
    domain_boundaries = list(
        map(lambda s: s.id, GTHeader._missing_domain_params(params))
    )
    assert len(domain_boundaries) == 0


def test_missing_domain_args_contains_all():
    domain_boundaries = list(map(lambda s: s.id, GTHeader._missing_domain_params([])))
    assert len(domain_boundaries) == len(backend._DOMAIN_ARGS)
    assert domain_boundaries == backend._DOMAIN_ARGS
