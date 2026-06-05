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
from typing import TYPE_CHECKING, Any

from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.common.topography.testcases import (
    from_file as from_file_topo,
    gauss3d as gauss_topo,
    jablonowski_williamson as jw_topo,
)
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs

@dataclasses.dataclass
class TopographyConfig:
    parameters: jw_topo.JablonowskiWilliamsonParameters | from_file_topo.FromFileParameters

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
            if data_path is None:
                return None
            return cls(
                parameters=from_file_topo.FromFileParameters(
                    data_path=data_path / fortran_config.SER_DATA_SUBDIR,
                ),
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        match testcase_nml.get("nh_test_name"):
            case "jabw" | "jabw_s":
                parameters = fortran_config.params_from_dict(
                    jw_topo.JablonowskiWilliamsonParameters, testcase_nml
                )
            case "gauss3D":
                raise NotImplementedError("Gauss3D topography is not yet implemented")
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(parameters=parameters)


def create(
    *,
    config: TopographyConfig,
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Create topography array by dispatching on the type of ``config.parameters``."""
    match config.parameters:
        case jw_topo.JablonowskiWilliamsonParameters():
            return jw_topo.jablonowski_williamson(parameters=config.parameters, grid_manager=grid_manager)
        case gauss_topo.Gauss3DParameters():
            return gauss_topo.gauss3d(parameters=config.parameters, grid_manager=grid_manager)
        case from_file_topo.FromFileParameters():
            return from_file_topo.read_from_file(parameters=config.parameters, grid_manager=grid_manager, backend=backend, exchange=exchange)
        case _:
            raise TypeError(f"Unknown topography parameters type: {type(config.parameters)!r}")
