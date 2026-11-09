# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import decomposer as decomp, definitions as decomp_defs
from icon4py.model.common.grid import grid_refinement, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    process_props,
)

from .. import utils
from . import utils as mpi_test_utils


_log = logging.getLogger(__name__)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "zone" in metafunc.fixturenames:
        params = [
            (dim, zone) for dim in dims.horizontal_dims() for zone in h_grid._get_zones_for_dim(dim)
        ]
        ids = [f"{dim.value}-{zone}" for dim, zone in params]
        metafunc.parametrize("dim,zone", params, ids=ids)
    elif "dim" in metafunc.fixturenames:
        ids = [dim.value for dim in dims.horizontal_dims()]
        metafunc.parametrize("dim", dims.horizontal_dims(), ids=ids)


@pytest.fixture
def domain(dim: gtx.Dimension, zone: h_grid.Zone) -> h_grid.Domain:
    return h_grid.domain(dim)(zone)


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_compute_domain_bounds(  # noqa: PLR0917 [too-many-positional-arguments]
    dim: gtx.Dimension,
    zone: h_grid.Zone,
    domain: h_grid.Domain,
    experiment: test_defs.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> None:
    if (
        process_props.is_single_rank()
        and experiment.description == test_defs.Experiments.EXCLAIM_APE
        and dim == dims.EdgeDim
    ):
        pytest.xfail(
            "end index data for single node APE are all 0 - re- serialization should fix that (patch%cells%end_index vs patch%cells%end_idx)"
        )

    ref_grid = grid_savepoint.construct_icon_grid(backend=backend, keep_skip_values=True)
    decomposition_info = grid_savepoint.construct_decomposition_info()
    refin_ctrl = {dim: grid_savepoint.refin_ctrl(dim) for dim in dims.horizontal_dims()}
    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)
    start_indices, end_indices = grid_refinement.compute_domain_bounds(
        dim, refin_ctrl, decomposition_info, array_ns=xp
    )
    if (
        experiment.description == test_defs.Experiments.GAUSS3D
        and dim == dims.EdgeDim
        and zone in (h_grid.Zone.LOCAL, h_grid.Zone.INTERIOR, h_grid.Zone.HALO)
    ):
        pytest.xfail(
            "ICON orders the halo with the negative refin_ctl values of the grid file and erases them "
            "afterwards (mo_setup_subdivision.f90), the serialized refin_ctl no longer has them."
        )

    ref_start_index = ref_grid.start_index(domain)
    ref_end_index = ref_grid.end_index(domain)
    computed_start = start_indices[domain]
    computed_end = end_indices[domain]
    _log.info(
        f"rank = {process_props.rank}/{process_props.comm_size}: domain={domain} : start = {computed_start} end = {computed_end} "
    )
    assert computed_start == ref_start_index, (
        f"rank={process_props.rank}/{process_props.comm_size} - experiment = {experiment.name}: start_index for {domain} does not match: is {computed_start}, expected {ref_start_index}"
    )
    assert computed_end == ref_end_index, (
        f"rank={process_props.rank}/{process_props.comm_size} - experiment = {experiment.name}: end_index for {domain} does not match: is {computed_end}, expected {ref_end_index}"
    )


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_bounds_decomposition(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
    experiment: test_defs.Experiment,
    dim: gtx.Dimension,
) -> None:
    file = dt_utils.get_grid_filepath(experiment.grid)
    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")

    grid_manager = mpi_test_utils.run_grid_manager_for_multi_rank(
        file=file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=model_backends.get_allocator(backend),
    )
    _log.info(
        f"rank = {process_props.rank} : {grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )

    decomposition_info = grid_manager.decomposition_info
    start_index = grid_manager.grid.start_index
    end_index = grid_manager.grid.end_index
    domain = h_grid.domain(dim)
    halo_levels = data_alloc.as_numpy(decomposition_info.halo_levels(dim))
    owner_mask = data_alloc.as_numpy(decomposition_info.owner_mask(dim))
    with_owned = grid_refinement.is_placed_with_owned(
        dim, halo_levels, data_alloc.as_numpy(grid_manager.grid.refinement_control[dim].ndarray)
    )
    num_with_owned = int(with_owned.sum())

    # owned points and the halo points ICON keeps with them come first
    assert with_owned[:num_with_owned].all()
    assert not owner_mask[num_with_owned:].any()
    assert start_index(domain(h_grid.Zone.LOCAL)) == 0
    assert end_index(domain(h_grid.Zone.LOCAL)) == num_with_owned
    assert end_index(domain(h_grid.Zone.INTERIOR)) == num_with_owned

    # then the other halo points, level by level
    position = num_with_owned
    for halo_domain in sorted(h_grid.get_halo_domains(dim), key=lambda d: d.zone.level):
        start, end = start_index(halo_domain), end_index(halo_domain)
        assert start == position
        assert (halo_levels[start:end] == halo_domain.zone.level).all()
        position = end
    assert position == end_index(domain(h_grid.Zone.END)) == halo_levels.size
    assert (
        decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.ALL).size
        == halo_levels.size
    )
