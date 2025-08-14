# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

import icon4py.model.common.grid.gridfile
import icon4py.model.testing.grid_utils as grid_utils
from icon4py.model.common import exceptions
from icon4py.model.common.decomposition import definitions, halo, mpi_decomposition
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid
from icon4py.model.testing import datatest_utils as dt_utils

from .. import utils


try:
    import mpi4py  # noqa F401:  import mpi4py to check for optional mpi dependency

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_props(caplog, processor_props):  # fixture
    caplog.set_level(logging.DEBUG)
    """dummy test to check setup"""
    assert processor_props.comm_size > 1


# TODO FIXME
@pytest.mark.xfail
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT)
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dims())
def test_start_end_index(
    caplog, backend, processor_props, grid_file, experiment, dim, icon_grid
):  # fixture
    caplog.set_level(logging.INFO)
    file = grid_utils.resolve_full_grid_file_name(grid_file)

    partitioner = halo.SimpleMetisDecomposer()
    manager = gm.GridManager(
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
        file,
        v_grid.VerticalGridConfig(1),
        run_properties=processor_props,
    )
    single_node_grid = gm.GridManager(
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
        file,
        v_grid.VerticalGridConfig(1),
        run_properties=definitions.get_processor_properties(definitions.SingleNodeRun()),
    ).grid
    with manager.set_decomposer(partitioner) as manage:
        manage(backend=backend, keep_skip_values=True)
        grid = manage.grid

    for domain in utils.global_grid_domains(dim):
        assert grid.start_index(domain) == single_node_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == single_node_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(processor_props):
    file = grid_utils.resolve_full_grid_file_name(dt_utils.R02B04_GLOBAL)
    manager = gm.GridManager(
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
        file,
        v_grid.VerticalGridConfig(1),
        run_properties=processor_props,
    )
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager.set_decomposer(halo.SingleNodeDecomposer())

    assert "Need a Decomposer for for multi" in e.value.args[0]
