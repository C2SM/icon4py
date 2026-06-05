# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from icon4py.model.common.topography.testcases import jablonowski_williamson as jw_topo
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


@dataclasses.dataclass
class TopographyConfig:
    parameters: jw_topo.JablonowskiWilliamsonTopographyParameters | None

    @classmethod
    def from_fortran_dict(
        cls,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        *,
        data_path: pathlib.Path | None = None,
    ) -> TopographyConfig | None:
        run_nml = atm_dict.get("run_nml", {})
        if not run_nml.get("ltestcase", False):
            return None

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        match testcase_nml.get("nh_test_name"):
            case "jabw" | "jabw_s":
                parameters = fortran_config.params_from_dict(
                    jw_topo.JablonowskiWilliamsonTopographyParameters, testcase_nml
                )
            case "gauss3D":
                raise NotImplementedError("Gauss3D topography is not yet implemented")
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(parameters=parameters)


def create(
    config: TopographyConfig,
    *,
    cell_lat: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Create topography array by dispatching on the type of ``config.parameters``."""
    match config.parameters:
        case jw_topo.JablonowskiWilliamsonTopographyParameters():
            return jw_topo.compute_topography(config.parameters, cell_lat=cell_lat)
        case None:
            raise TypeError(
                "TopographyConfig.parameters is None; no analytical topography available"
            )
        case _:
            raise TypeError(f"Unknown topography parameters type: {type(config.parameters)!r}")
