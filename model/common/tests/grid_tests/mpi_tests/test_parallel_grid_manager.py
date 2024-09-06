# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

from icon4py.model.common.decomposition import halo
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)

from .. import utils


try:
    import mpi4py  # noqa F401:  import mpi4py to check for optional mpi dependency
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)


# mpi marker meses up mpi initialization
# @pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_props(caplog, processor_props):  # noqa: F811  # fixture
    caplog.set_level(logging.DEBUG)
    """dummy test to check setup"""
    assert processor_props.comm_size > 1


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT)
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_start_end_index(caplog, processor_props, grid_file, experiment, dim, icon_grid):  # noqa: F811  # fixture
    caplog.set_level(logging.INFO)
    file = utils.resolve_file_from_gridfile_name(grid_file)
    limited_area = experiment == dt_utils.REGIONAL_EXPERIMENT
    partitioner = halo.SimpleMetisDecomposer()
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), file, v_grid.VerticalGridConfig(1)
    )
    with manager.with_decomposer(partitioner, processor_props) as manage:
        manage(limited_area=limited_area)
        grid = manage.grid

    for domain in utils.global_grid_domains(dim):
        assert grid.start_index(domain) == utils.single_node_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == utils.single_node_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"
