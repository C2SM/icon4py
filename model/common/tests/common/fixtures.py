# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import random
from collections.abc import Generator

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common.grid import geometry, geometry_attributes, gridfile, vertical
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import serialbox
from icon4py.model.testing.definitions import construct_metrics_config
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    damping_height,
    data_provider,
    decomposition,
    decomposition_info,
    definitions,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    linit,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    stretch_factor,
    topography_savepoint,
)


@pytest.fixture
def random_name() -> str:
    return "test" + str(random.randint(0, 100000))


@pytest.fixture
def test_path(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
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


@pytest.fixture
def geometry_from_savepoint(
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
    decomposition_info: decomposition.DecompositionInfo,
    processor_props: decomposition.ProcessProperties,
) -> Generator[geometry.GridGeometry]:
    grid = grid_savepoint.construct_icon_grid(backend, with_repeated_index=False)
    coordinates = grid_savepoint.coordinates()
    extra_fields = {
        gridfile.GeometryName.CELL_AREA: grid_savepoint.cell_areas(),
        gridfile.GeometryName.EDGE_LENGTH: grid_savepoint.primal_edge_length(),
        gridfile.GeometryName.DUAL_EDGE_LENGTH: grid_savepoint.dual_edge_length(),
        gridfile.GeometryName.EDGE_CELL_DISTANCE: grid_savepoint.edge_cell_length(),
        gridfile.GeometryName.EDGE_VERTEX_DISTANCE: grid_savepoint.edge_vert_length(),
        gridfile.GeometryName.DUAL_AREA: grid_savepoint.vertex_dual_area(),
        gridfile.GeometryName.TANGENT_ORIENTATION: grid_savepoint.tangent_orientation(),
        gridfile.GeometryName.CELL_NORMAL_ORIENTATION: grid_savepoint.edge_orientation(),
        gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: grid_savepoint.vertex_edge_orientation(),
    }

    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    global_reductions = decomposition.create_reduction(processor_props)
    grid_geometry = geometry.GridGeometry(
        grid=grid,
        decomposition_info=decomposition_info,
        backend=backend,
        metadata=geometry_attributes.attrs,
        coordinates=coordinates,
        extra_fields=extra_fields,
        exchange=exchange,
        global_reductions=global_reductions,
    )
    yield grid_geometry


@pytest.fixture
def interpolation_factory_from_savepoint(
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
    decomposition_info: decomposition.DecompositionInfo,
    processor_props: decomposition.ProcessProperties,
    geometry_from_savepoint: geometry.GridGeometry,
) -> Generator[interpolation_factory.InterpolationFieldsFactory]:
    geometry_source = geometry_from_savepoint
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    intp_factory = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry_source.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        cell_geometry=grid_savepoint.construct_cell_geometry(),
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=exchange,
    )
    yield intp_factory


@pytest.fixture
def metrics_factory_from_savepoint(
    backend: gtx_typing.Backend,
    grid_savepoint: serialbox.IconGridSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    decomposition_info: decomposition.DecompositionInfo,
    processor_props: decomposition.ProcessProperties,
    geometry_from_savepoint: geometry.GridGeometry,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
) -> Generator[metrics_factory.MetricsFieldsFactory]:
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    global_reductions = decomposition.create_reduction(processor_props)
    geometry_source = geometry_from_savepoint
    interpolation_field_source = interpolation_factory_from_savepoint
    topography = topography_savepoint.topo_c()
    (
        lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type,
    ) = construct_metrics_config(experiment)
    vertical_config = vertical.VerticalGridConfig(
        geometry_source.grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_grid = vertical.VerticalGrid(
        vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
    )
    factory = metrics_factory.MetricsFieldsFactory(
        grid=geometry_source.grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        topography=topography,
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        exner_expol=exner_expol,
        vwind_offctr=vwind_offctr,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    yield factory
