# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import operator

import numpy as np
import pytest
from gt4py import next as gtx
from gt4py.next import common as gtx_common, typing as gtx_typing

from icon4py.model.common import dimension as dims, exceptions, model_backends
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomp_defs,
    mpi_decomposition,
)
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes,
    grid_manager as gm,
    gridfile,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    experiment,
    processor_props,
    topography_savepoint,
)

from . import utils


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(
    processor_props: decomp_defs.ProcessProperties,
    experiment: test_defs.Experiment,
) -> None:
    if experiment == test_defs.Experiments.MCH_CH_R04B09:
        pytest.xfail("Limited-area grids not yet supported")

    file = grid_utils.resolve_full_grid_file_name(experiment.grid)
    manager = gm.GridManager(
        grid_file=file,
        config=v_grid.VerticalGridConfig(num_levels=utils.NUM_LEVELS),
        offset_transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            allocator=None,
            run_properties=processor_props,
            decomposer=decomp.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


def _get_neighbor_tables(grid: base.Grid) -> dict:
    return {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }


def gather_field(field: np.ndarray, props: decomp_defs.ProcessProperties) -> tuple:
    constant_dims = tuple(field.shape[1:])
    _log.info(f"gather_field on rank={props.rank} - gathering field of local shape {field.shape}")
    constant_length = functools.reduce(operator.mul, constant_dims, 1)
    local_sizes = np.array(props.comm.gather(field.size, root=0))
    if props.rank == 0:
        recv_buffer = np.empty(np.sum(local_sizes), dtype=field.dtype)
        _log.info(
            f"gather_field on rank = {props.rank} - setup receive buffer with size {sum(local_sizes)} on rank 0"
        )
    else:
        recv_buffer = None

    props.comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    if props.rank == 0:
        local_first_dim = tuple(sz // constant_length for sz in local_sizes)
        _log.info(
            f" gather_field on rank = 0: computed local dims {local_first_dim} - constant dims {constant_dims}"
        )
        gathered_field = recv_buffer.reshape((-1, *constant_dims))  # type: ignore [union-attr]
    else:
        gathered_field = None
        local_first_dim = field.shape
    return local_first_dim, gathered_field


def check_local_global_field(
    decomposition_info: decomp_defs.DecompositionInfo,
    processor_props: decomp_defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
    check_halos: bool,
) -> None:
    _log.info(
        f" rank= {processor_props.rank}/{processor_props.comm_size}----exchanging field of main dim {dim}"
    )
    assert (
        local_field.shape[0]
        == decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.ALL).shape[
            0
        ]
    )

    # Compare halo against global reference field
    if check_halos:
        np.testing.assert_allclose(
            global_reference_field[
                decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.HALO)
            ],
            local_field[
                decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.HALO)
            ],
            atol=1e-9,
            verbose=True,
        )

    # Compare owned local field, excluding halos, against global reference
    # field, by gathering owned entries to the first rank. This ensures that in
    # total we have the full global field distributed on all ranks.
    owned_entries = local_field[
        decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props)

    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED),
        processor_props,
    )

    if processor_props.rank == 0:
        _log.info(f"rank = {processor_props.rank}: asserting gathered fields: ")

        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match:  {dim} {gathered_sizes} - {global_index_sizes}"
        _log.info(
            f"rank = {processor_props.rank}: Checking field size on dim ={dim}: --- gathered sizes {gathered_sizes} = {sum(gathered_sizes)}"
        )
        _log.info(
            f"rank = {processor_props.rank}:                      --- gathered field has size {gathered_sizes}"
        )
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)  # type: ignore [attr-defined]
        sorted_[gathered_global_indices] = gathered_field
        _log.info(
            f" rank = {processor_props.rank}: SHAPES: global reference field {global_reference_field.shape}, gathered = {gathered_field.shape}"
        )

        # TODO(msimberg): Is this true? Not true for RBF itnerpolation... why?
        # We expect an exact match, since the starting point is the same (grid
        # file) and we are doing the exact same computations in single rank and
        # multi rank mode.
        np.testing.assert_allclose(sorted_, global_reference_field, atol=1e-9, verbose=True)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        geometry_attributes.CELL_AREA,
        geometry_attributes.CELL_CENTER_X,
        geometry_attributes.CELL_CENTER_Y,
        geometry_attributes.CELL_CENTER_Z,
        geometry_attributes.CELL_LAT,
        geometry_attributes.CELL_LON,
        geometry_attributes.CELL_NORMAL_ORIENTATION,
        geometry_attributes.CORIOLIS_PARAMETER,
        geometry_attributes.DUAL_AREA,
        geometry_attributes.DUAL_EDGE_LENGTH,
        geometry_attributes.EDGE_AREA,
        geometry_attributes.EDGE_CELL_DISTANCE,
        geometry_attributes.EDGE_CENTER_X,
        geometry_attributes.EDGE_CENTER_Y,
        geometry_attributes.EDGE_CENTER_Z,
        geometry_attributes.EDGE_DUAL_U,
        geometry_attributes.EDGE_DUAL_V,
        geometry_attributes.EDGE_LAT,
        geometry_attributes.EDGE_LENGTH,
        geometry_attributes.EDGE_LON,
        geometry_attributes.EDGE_NORMAL_CELL_U,
        geometry_attributes.EDGE_NORMAL_CELL_V,
        geometry_attributes.EDGE_NORMAL_U,
        geometry_attributes.EDGE_NORMAL_V,
        geometry_attributes.EDGE_NORMAL_VERTEX_U,
        geometry_attributes.EDGE_NORMAL_VERTEX_V,
        geometry_attributes.EDGE_NORMAL_X,
        geometry_attributes.EDGE_NORMAL_Y,
        geometry_attributes.EDGE_NORMAL_Z,
        geometry_attributes.EDGE_TANGENT_CELL_U,
        geometry_attributes.EDGE_TANGENT_CELL_V,
        geometry_attributes.EDGE_TANGENT_VERTEX_U,
        geometry_attributes.EDGE_TANGENT_VERTEX_V,
        geometry_attributes.EDGE_TANGENT_X,
        geometry_attributes.EDGE_TANGENT_Y,
        geometry_attributes.EDGE_TANGENT_Z,
        geometry_attributes.EDGE_VERTEX_DISTANCE,
        geometry_attributes.TANGENT_ORIENTATION,
        geometry_attributes.VERTEX_EDGE_ORIENTATION,
        geometry_attributes.VERTEX_LAT,
        geometry_attributes.VERTEX_LON,
        geometry_attributes.VERTEX_VERTEX_LENGTH,  # TODO(msimberg): Also inverse?
        geometry_attributes.VERTEX_X,
        geometry_attributes.VERTEX_Y,
        geometry_attributes.VERTEX_Z,
    ],
)
def test_geometry_fields_compare_single_multi_rank(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    if experiment == test_defs.Experiments.MCH_CH_R04B09:
        pytest.xfail("Limited-area grids not yet supported")

    # TODO(msimberg): Add fixtures for single/multi-rank
    # grid/geometry/interpolation/metrics factories.
    file = grid_utils.resolve_full_grid_file_name(experiment.grid)
    _log.info(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_rank_grid_manager = utils.run_grid_manager_for_single_rank(file)
    single_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_rank_grid_manager.grid,
        coordinates=single_rank_grid_manager.coordinates,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        extra_fields=single_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    _log.info(
        f"rank = {processor_props.rank} : single node grid has size {single_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_grid_manager = utils.run_grid_manager_for_multi_rank(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
    )
    _log.info(
        f"rank = {processor_props.rank} : {multi_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=multi_rank_grid_manager.grid,
        coordinates=multi_rank_grid_manager.coordinates,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        extra_fields=multi_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
        exchange=decomp_defs.create_exchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
        global_reductions=decomp_defs.create_reduction(processor_props),
    )

    dim = single_rank_geometry.get(attrs_name).domain.dims[0]
    field_ref = single_rank_geometry.get(attrs_name).asnumpy()
    field = multi_rank_geometry.get(attrs_name).asnumpy()

    check_halos = True
    check_local_global_field(
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        processor_props=processor_props,
        dim=dim,
        global_reference_field=field_ref,
        local_field=field,
        check_halos=check_halos,
    )

    _log.info(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        interpolation_attributes.GEOFAC_DIV,
        interpolation_attributes.GEOFAC_ROT,
        interpolation_attributes.C_BLN_AVG,
        interpolation_attributes.RBF_VEC_COEFF_C1,
        interpolation_attributes.RBF_VEC_COEFF_C2,
        interpolation_attributes.RBF_VEC_COEFF_E,
        interpolation_attributes.RBF_VEC_COEFF_V1,
        interpolation_attributes.RBF_VEC_COEFF_V2,
    ],
)
def test_interpolation_fields_compare_single_multi_rank(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    if experiment == test_defs.Experiments.MCH_CH_R04B09:
        pytest.xfail("Limited-area grids not yet supported")

    file = grid_utils.resolve_full_grid_file_name(experiment.grid)
    _log.info(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_rank_grid_manager = utils.run_grid_manager_for_single_rank(file)
    single_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_rank_grid_manager.grid,
        coordinates=single_rank_grid_manager.coordinates,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        extra_fields=single_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    single_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=single_rank_grid_manager.grid,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        geometry_source=single_rank_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    _log.info(
        f"rank = {processor_props.rank} : single node grid has size {single_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_grid_manager = utils.run_grid_manager_for_multi_rank(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
    )
    _log.info(
        f"rank = {processor_props.rank} : {multi_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=multi_rank_grid_manager.grid,
        coordinates=multi_rank_grid_manager.coordinates,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        extra_fields=multi_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
        exchange=decomp_defs.create_exchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
        global_reductions=decomp_defs.create_reduction(processor_props),
    )
    multi_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_rank_grid_manager.grid,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        geometry_source=multi_rank_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.create_exchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
    )

    dim = single_rank_geometry.get(attrs_name).domain.dims[0]
    field_ref = single_rank_interpolation.get(attrs_name).asnumpy()
    field = multi_rank_interpolation.get(attrs_name).asnumpy()

    check_halos = True
    check_local_global_field(
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        processor_props=processor_props,
        dim=dim,
        global_reference_field=field_ref,
        local_field=field,
        check_halos=check_halos,
    )

    _log.info(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("attrs_name", [metrics_attributes.DDXT_Z_HALF_E])
def test_metrics_fields_compare_single_multi_rank(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    if experiment == test_defs.Experiments.MCH_CH_R04B09:
        pytest.xfail("Limited-area grids not yet supported")

    file = grid_utils.resolve_full_grid_file_name(experiment.grid)

    (
        lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type,
        thslp_zdiffu,
        thhgtd_zdiffu,
    ) = test_defs.construct_metrics_config(experiment)
    vertical_config = v_grid.VerticalGridConfig(
        experiment.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    # TODO(msimberg): Dummy vct_a? Taken from test_io.py.
    xp = data_alloc.import_array_ns(backend)
    allocator = model_backends.get_allocator(backend)
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.num_levels + 1),
            allocator=allocator,
        ),
        vct_b=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.num_levels + 1),
            allocator=allocator,
        ),
    )

    _log.info(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_rank_grid_manager = utils.run_grid_manager_for_single_rank(file, experiment.num_levels)
    single_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_rank_grid_manager.grid,
        coordinates=single_rank_grid_manager.coordinates,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        extra_fields=single_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    single_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=single_rank_grid_manager.grid,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        geometry_source=single_rank_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    single_rank_metrics = metrics_factory.MetricsFieldsFactory(
        grid=single_rank_geometry.grid,
        vertical_grid=vertical_grid,
        decomposition_info=single_rank_grid_manager.decomposition_info,
        geometry_source=single_rank_geometry,
        # TODO(msimberg): Valid dummy topography?
        topography=(
            gtx.as_field(
                (dims.CellDim,),
                xp.zeros(single_rank_geometry.grid.num_cells),
                allocator=allocator,
            )
        ),
        interpolation_source=single_rank_interpolation,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    _log.info(
        f"rank = {processor_props.rank} : single node grid has size {single_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_grid_manager = utils.run_grid_manager_for_multi_rank(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
        num_levels=experiment.num_levels,
    )
    _log.info(
        f"rank = {processor_props.rank} : {multi_rank_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_geometry = geometry.GridGeometry(
        backend=backend,
        grid=multi_rank_grid_manager.grid,
        coordinates=multi_rank_grid_manager.coordinates,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        extra_fields=multi_rank_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
        exchange=decomp_defs.create_exchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
        global_reductions=decomp_defs.create_reduction(processor_props),
    )
    multi_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_rank_grid_manager.grid,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        geometry_source=multi_rank_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.create_exchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
    )
    multi_rank_metrics = metrics_factory.MetricsFieldsFactory(
        grid=multi_rank_geometry.grid,
        vertical_grid=vertical_grid,
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        geometry_source=multi_rank_geometry,
        # TODO(msimberg): Valid dummy topography?
        topography=(
            gtx.as_field(
                (dims.CellDim,),
                xp.zeros(multi_rank_geometry.grid.num_cells),
                allocator=allocator,
            )
        ),
        interpolation_source=multi_rank_interpolation,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        exchange=mpi_decomposition.GHexMultiNodeExchange(
            processor_props, multi_rank_grid_manager.decomposition_info
        ),
    )

    dim = single_rank_geometry.get(attrs_name).domain.dims[0]
    field_ref = single_rank_metrics.get(attrs_name).asnumpy()
    field = multi_rank_metrics.get(attrs_name).asnumpy()

    check_halos = True
    check_local_global_field(
        decomposition_info=multi_rank_grid_manager.decomposition_info,
        processor_props=processor_props,
        dim=dim,
        global_reference_field=field_ref,
        local_field=field,
        check_halos=check_halos,
    )

    _log.info(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_validate_skip_values_in_distributed_connectivities(
    processor_props: decomp_defs.ProcessProperties,
    experiment: test_defs.Experiment,
) -> None:
    if experiment == test_defs.Experiments.MCH_CH_R04B09:
        pytest.xfail("Limited-area grids not yet supported")

    file = grid_utils.resolve_full_grid_file_name(experiment.grid)
    multi_rank_grid_manager = utils.run_grid_manager_for_multi_rank(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
    )
    distributed_grid = multi_rank_grid_manager.grid
    for k, c in distributed_grid.connectivities.items():
        if gtx_common.is_neighbor_connectivity(c):
            skip_values_in_table = np.count_nonzero(c.asnumpy() == c.skip_value)
            found_skips = skip_values_in_table > 0
            assert (
                found_skips == (c.skip_value is not None)
            ), f"rank={processor_props.rank} / {processor_props.comm_size}: {k} - # of skip values found in table = {skip_values_in_table},  skip value is {c.skip_value}"
            if skip_values_in_table > 0:
                dim = gtx.Dimension(k, gtx.DimensionKind.LOCAL)
                assert (
                    dim in icon.CONNECTIVITIES_ON_BOUNDARIES
                    or dim in icon.CONNECTIVITIES_ON_PENTAGONS
                ), f"rank={processor_props.rank} / {processor_props.comm_size}: {k} has skip found in table, expected none"


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", [test_defs.Grids.MCH_CH_R04B09_DSL])
def test_limited_area_raises(
    processor_props: decomp_defs.ProcessProperties,
    grid: test_defs.GridDescription,
) -> None:
    with pytest.raises(
        NotImplementedError, match="Limited-area grids are not supported in distributed runs"
    ):
        _ = utils.run_grid_manager_for_multi_rank(
            file=grid_utils.resolve_full_grid_file_name(grid),
            run_properties=processor_props,
            decomposer=decomp.MetisDecomposer(),
        )
