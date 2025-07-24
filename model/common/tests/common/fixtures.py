# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import random

import pytest

from icon4py.model.testing.fixtures import (  # noqa: F401
    backend,
    connectivities_as_numpy,
    decomposition_info,
    experiment,
    grid,
)
from icon4py.model.testing.fixtures.datatest import (  # noqa: F401
    damping_height,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    metrics_savepoint,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
)


@pytest.fixture
def random_name() -> str:
    return "test" + str(random.randint(0, 100000))


@pytest.fixture
def test_path(tmp_path):
    base_path = tmp_path.joinpath("io_tests")
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)
    yield base_path
    _delete_recursive(base_path)


def _delete_recursive(p: pathlib.Path) -> None:
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _delete_recursive(child)
    p.rmdir()
