# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.driver.testcases import gauss3d
from icon4py.model.testing import datatest_utils as dt_utils, definitions, test_utils
from icon4py.model.testing.fixtures.datatest import backend

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import pathlib

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, rank",
    [
        (dt_utils.GAUSS3D_EXPERIMENT, 0),
    ],
)
def test_gauss3d_initial_condition(
    experiment: definitions.Experiment,
    ranked_data_path: pathlib.Path,
    backend: gtx_typing.Backend,
    rank: int,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    icon_grid: base_grid.Grid,
) -> None:
    edge_geometry = grid_savepoint.construct_edge_geometry()

    (
        _,
        _,
        _,
        _,
        _,
        prognostic_state_now,
        _,
    ) = gauss3d.model_initialization_gauss3d(
        icon_grid,
        edge_geometry,
        ranked_data_path.joinpath(f"{experiment}/ser_data"),
        backend,
        rank,
    )

    # only verifying those assigned in the IC rather than all (at least for now)
    prognostics_reference_savepoint = data_provider.from_savepoint_prognostics_initial()
    assert test_utils.dallclose(
        prognostic_state_now.rho.asnumpy(),
        prognostics_reference_savepoint.rho_now().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.exner.asnumpy(),
        prognostics_reference_savepoint.exner_now().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        prognostics_reference_savepoint.theta_v_now().asnumpy(),
    )
