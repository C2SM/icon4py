# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common.constants import RayleighType
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.initialization import jablonowski_williamson_topography as topology
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import definitions, grid_utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


@pytest.fixture(
    scope="session",
    params=[definitions.Grids.R19_B07_MCH_LOCAL, definitions.Grids.R02B04_GLOBAL],
)
def benchmark_grid(request: pytest.FixtureRequest) -> definitions.GridDescription:
    """Default parametrization for benchmark testing.

    The default parametrization is often overwritten for specific tests."""
    return request.param


@pytest.fixture(scope="session")
def grid_manager(
    benchmark_grid: definitions.GridDescription,
    backend: gtx_typing.Backend | None,
) -> gm.GridManager:
    grid_manager = grid_utils.get_grid_manager_from_identifier(
        benchmark_grid, num_levels=80, keep_skip_values=True, backend=backend
    )
    return grid_manager


@pytest.fixture(
    scope="session",
)
def geometry_field_source(
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend | None,
) -> Generator[grid_geometry.GridGeometry, None, None]:
    mesh = grid_manager.grid

    decomposition_info = grid_utils.construct_decomposition_info(mesh, backend)

    geometry_field_source = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=grid_manager.coordinates,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_meta.attrs,
    )
    yield geometry_field_source
    del geometry_field_source


@pytest.fixture(
    scope="session",
)
def interpolation_field_source(
    grid_manager: gm.GridManager,
    geometry_field_source: grid_geometry.GridGeometry,
    backend: gtx_typing.Backend | None,
) -> Generator[interpolation_factory.InterpolationFieldsFactory, None, None]:
    mesh = grid_manager.grid

    decomposition_info = grid_utils.construct_decomposition_info(mesh, backend)

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        grid=mesh,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
    )
    yield interpolation_field_source
    del interpolation_field_source


@pytest.fixture(
    scope="session",
)
def metrics_field_source(
    grid_manager: gm.GridManager,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    backend: gtx_typing.Backend | None,
) -> Generator[metrics_factory.MetricsFieldsFactory, None, None]:
    mesh = grid_manager.grid

    decomposition_info = grid_utils.construct_decomposition_info(mesh, backend)

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
    )

    topo_c = topology.jablonowski_williamson_topography(
        cell_lat=cell_geometry.cell_center_lat.ndarray,
        u0=35.0,
        backend=backend,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),  # type: ignore[arg-type]  # NDArrayObject is not exported from gt4py
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        rayleigh_type=RayleighType.KLEMP,
        rayleigh_coeff=5.0,
        exner_expol=0.333,
        vwind_offctr=0.2,
    )
    yield metrics_field_source
    del metrics_field_source
