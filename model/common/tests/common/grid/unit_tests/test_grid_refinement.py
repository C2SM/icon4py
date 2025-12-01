# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import grid_refinement as refinement, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import definitions as test_defs, grid_utils, serialbox
from icon4py.model.testing.fixtures import backend, cpu_allocator

from .. import utils
from ..fixtures import (
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    processor_props,
    ranked_data_path,
)


_FALLBACK_FAIL = (-10, -10)


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize(
    "grid_file, expected",
    [
        (test_defs.Grids.R02B04_GLOBAL, False),
        (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, True),
    ],
)
def test_is_local_area_grid_for_grid_files(
    grid_file: test_defs.GridDescription,
    expected: bool,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend | None,
) -> None:
    grid = grid_utils.get_grid_manager_from_identifier(
        grid_file, 1, True, model_backends.get_allocator(backend)
    ).grid
    xp = data_alloc.array_ns(device_utils.is_cupy_device(backend))
    refinement_field = grid.refinement_control[dim]
    limited_area = refinement.is_limited_area_grid(refinement_field.ndarray, array_ns=xp)
    assert isinstance(limited_area, bool)
    assert expected == limited_area


cell_bounds: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 629),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (629, 1244),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (1244, 1843),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1843, 2424),
    h_grid.Zone.NUDGING: (2424, 2989),
    h_grid.Zone.INTERIOR: (2989, 10700),
    h_grid.Zone.LOCAL: (0, 10700),
    h_grid.Zone.END: (10700, 10700),
    h_grid.Zone.HALO: (10700, 10700),
    h_grid.Zone.HALO_LEVEL_2: (10700, 10700),
}
edge_bounds: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 318),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (318, 947),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (947, 1258),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1258, 1873),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5: (1873, 2177),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6: (2177, 2776),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7: (2776, 3071),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8: (3071, 3652),
    h_grid.Zone.NUDGING: (3652, 3938),
    h_grid.Zone.NUDGING_LEVEL_2: (3938, 4503),
    h_grid.Zone.INTERIOR: (4503, 16209),
    h_grid.Zone.LOCAL: (0, 16209),
    h_grid.Zone.END: (16209, 16209),
    h_grid.Zone.HALO: (16209, 16209),
    h_grid.Zone.HALO_LEVEL_2: (16209, 16209),
}
vertex_bounds: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 318),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (318, 629),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (629, 933),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (933, 1228),
    h_grid.Zone.NUDGING: (1228, 1514),
    h_grid.Zone.INTERIOR: (1514, 5510),
    h_grid.Zone.LOCAL: (0, 5510),
    h_grid.Zone.END: (5510, 5510),
    h_grid.Zone.HALO: (5510, 5510),
    h_grid.Zone.HALO_LEVEL_2: (5510, 5510),
}


@pytest.mark.parametrize(
    "dim, expected",
    [(dims.CellDim, cell_bounds), (dims.EdgeDim, edge_bounds), (dims.VertexDim, vertex_bounds)],
)
def test_compute_start_index_for_limited_area_grid(
    dim: gtx.Dimension,
    expected: dict[h_grid.Zone, tuple[int, int]],
    cpu_allocator: gtx_typing.FieldBufferAllocationUtil,
) -> None:
    grid_manager = grid_utils.get_grid_manager_from_identifier(
        test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, 1, True, cpu_allocator
    )
    grid = grid_manager.grid
    refinement_field = grid.refinement_control
    decomposition_info = grid_manager.decomposition_info
    start_index, end_index = refinement.compute_domain_bounds(
        dim, refinement_field, decomposition_info=decomposition_info, array_ns=np
    )

    for d, v in start_index.items():
        expected_value = expected.get(d.zone, _FALLBACK_FAIL)[0]
        assert (
            v == expected_value
        ), f"Expected start index {expected_value} for domain = {d} , but got {v}"

    for d, v in end_index.items():
        expected_value = expected.get(d.zone, _FALLBACK_FAIL)[1]
        assert (
            v == expected_value
        ), f"Expected end index {expected_value} for domain = {d} , but got {v}"


@pytest.mark.parametrize("file", (test_defs.Grids.R02B04_GLOBAL,))
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_compute_domain_bounds_for_global_grid(
    file: test_defs.GridDescription,
    dim: gtx.Dimension,
    cpu_allocator: gtx_typing.FieldBufferAllocationUtil,
) -> None:
    grid_manager = grid_utils.get_grid_manager_from_identifier(file, 1, True, cpu_allocator)
    grid = grid_manager.grid
    refinement_fields = grid.refinement_control
    decomposition_info = grid_manager.decomposition_info
    start_index, end_index = refinement.compute_domain_bounds(
        dim, refinement_fields, decomposition_info, array_ns=np
    )
    for k, v in start_index.items():
        assert isinstance(v, gtx.int32)
        if k.zone.is_halo() or k.zone is h_grid.Zone.END:
            assert (
                v == grid.size[dim]
            ), f"Expected start index '{grid.size[dim]}' for '{dim}' in zone '{k.zone}', but got '{v}'"
        else:
            assert v == 0, f"Expected start index '0' for {dim} in {k.zone}, but got '{v}'"

    for k, v in end_index.items():
        assert isinstance(v, gtx.int32)
        if k.zone.is_lateral_boundary() or k.zone.is_nudging():
            assert v == 0, f"Expected end index '0' for '{dim}' in zone '{k.zone}', but got '{v}'"
        else:
            assert (
                v == grid.size[k.dim]
            ), f"Expected end index '{grid.size[k.dim]}' for {dim} in {k.zone}, but got '{v}'"


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.datatest
def test_start_end_index(
    dim: gtx.Dimension,
    experiment: test_defs.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
) -> None:
    ref_grid = grid_savepoint.construct_icon_grid(None, keep_skip_values=True)
    decomposition_info = grid_savepoint.construct_decomposition_info()
    refin_ctrl = {dim: grid_savepoint.refin_ctrl(dim) for dim in utils.main_horizontal_dims()}
    start_indices, end_indices = refinement.compute_domain_bounds(
        dim, refin_ctrl, decomposition_info
    )
    for domain in h_grid.get_domains_for_dim(dim):
        ref_start_index = ref_grid.start_index(domain)
        ref_end_index = ref_grid.end_index(domain)
        computed_start = start_indices[domain]
        computed_end = end_indices[domain]
        assert (
            computed_start == ref_start_index
        ), f" experiment = {experiment.name}: start_index for {domain} does not match: is {computed_start}, expected {ref_start_index}"
        assert (
            computed_end == ref_end_index
        ), f"experiment = {experiment.name}: end_index for {domain} does not match: is {computed_end}, expected {ref_end_index}"
