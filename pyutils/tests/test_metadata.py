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

import re
from typing import List

import pytest

from icon4py.pyutils.icon4pygen import (
    get_fvprog,
    import_definition,
    scan_for_offsets,
)
from icon4py.pyutils.metadata import format_metadata
from icon4py.testutils.utils import get_stencil_module_path


def get_stencil_metadata(stencil_module: str, stencil_name: str) -> str:
    fencil = import_definition(get_stencil_module_path(stencil_module, stencil_name))
    fvprog = get_fvprog(fencil)
    chains = scan_for_offsets(fvprog)
    return format_metadata(fvprog, chains)


def parse_tabulated(tabulated: str, col: str) -> List[str]:
    tabulated_list = [re.split(r"\s\s+", line) for line in tabulated.splitlines()]
    offsets = tabulated_list.pop(0)

    parsed = [
        dict(zip(["name", "type", "io"], tabulated_list[i]))
        for i in range(len(tabulated_list))
    ]

    if col == "offsets":
        return offsets[0].split(", ")

    return [arg[col] for arg in parsed]


@pytest.mark.parametrize(
    (
        "stencil_module",
        "stencil_name",
        "exp_offsets",
        "exp_names",
        "exp_types",
        "exp_in",
        "exp_out",
        "exp_inout",
    ),
    [
        (
            "atm_dyn_iconam",
            "mo_nh_diffusion_stencil_14",
            {"C2CE", "C2E"},
            ["z_nabla2_e", "geofac_div", "z_temp"],
            [
                "Field[[Edge, K], dtype=float64]",
                "Field[[CE], dtype=float64]",
                "Field[[Cell, K], dtype=float64]",
            ],
            2,
            1,
            0,
        ),
        (
            "atm_dyn_iconam",
            "mo_solve_nonhydro_stencil_29",
            {""},
            ["grf_tend_vn", "vn_now", "vn_new", "dtime"],
            [
                "Field[[Edge, K], dtype=float64]",
                "Field[[Edge, K], dtype=float64]",
                "Field[[Edge, K], dtype=float64]",
                "Field[[], dtype=float64]",
            ],
            3,
            1,
            0,
        ),
        (
            "atm_dyn_iconam",
            "mo_nh_diffusion_stencil_06",
            {""},
            ["z_nabla2_e", "area_edge", "vn", "fac_bdydiff_v"],
            [
                "Field[[Edge, K], dtype=float64]",
                "Field[[Edge], dtype=float64]",
                "Field[[Edge, K], dtype=float64]",
                "Field[[], dtype=float64]",
            ],
            3,
            0,
            1,
        ),
    ],
)
def test_tabulation(
    stencil_module,
    stencil_name,
    exp_offsets,
    exp_names,
    exp_types,
    exp_in,
    exp_out,
    exp_inout,
):
    tabulated = get_stencil_metadata(stencil_module, stencil_name)
    parsed_names = parse_tabulated(tabulated, "name")
    parsed_types = parse_tabulated(tabulated, "type")
    parsed_io = parse_tabulated(tabulated, "io")
    parsed_offsets = set(parse_tabulated(tabulated, "offsets"))

    assert len(parsed_offsets - exp_offsets) == 0
    assert parsed_names == exp_names
    assert parsed_types == exp_types
    assert (
        parsed_io.count("in") == exp_in
        and parsed_io.count("out") == exp_out
        and parsed_io.count("inout") == exp_inout
    )
