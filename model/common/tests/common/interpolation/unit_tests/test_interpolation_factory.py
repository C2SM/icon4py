# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import backend as gtx_backend

import icon4py.model.common.states.factory as factory
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
    rbf_interpolation as rbf,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    serialbox as sb,
    test_utils as test_helpers,
)
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    decomposition_info,
    experiment,
)
from icon4py.model.testing.fixtures.datatest import (
    download_ser_data,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)

from ...fixtures import grid_file


V2E_SIZE = 6

C2E_SIZE = 3
E2C_SIZE = 2


interpolation_factories = {}

cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_raises_error_on_unknown_field(grid_file, experiment, backend, decomposition_info):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=attrs.attrs,
    )
    with pytest.raises(ValueError) as error:
        interpolation_source.get("foo", factory.RetrievalType.METADATA)
        assert "unknown field" in error.value


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_c_lin_e(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.c_lin_e()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.C_LIN_E)
    assert field.shape == (grid.num_edges, E2C_SIZE)
    assert test_helpers.dallclose(field.asnumpy(), field_ref.asnumpy(), rtol=rtol)


def _get_interpolation_factory(
    backend: gtx_backend.Backend | None, experiment: str, grid_file: str
) -> interpolation_factory.InterpolationFieldsFactory:
    registry_key = "_".join((experiment, data_alloc.backend_name(backend)))
    factory = interpolation_factories.get(registry_key)
    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)

        factory = interpolation_factory.InterpolationFieldsFactory(
            grid=geometry.grid,
            decomposition_info=geometry._decomposition_info,
            geometry_source=geometry,
            backend=backend,
            metadata=attrs.attrs,
        )
        interpolation_factories[registry_key] = factory
    return factory


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 1e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-12),
    ],
)
@pytest.mark.datatest
def test_get_geofac_div(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_div()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.level("integration")
## FIXME: does not validate
#   -> connectivity order between reference from serialbox and computed value is different
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_grdiv(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_grdiv()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_GRDIV)
    assert field.shape == (grid.num_edges, 5)
    # FIXME: e2c2e constructed from grid file has different ordering than the serialized one
    assert_reordered(field.asnumpy(), field_ref.asnumpy(), rtol=rtol)


def assert_reordered(val: np.ndarray, ref: np.ndarray, **kwargs):
    assert val.shape == ref.shape, f"arrays do not have the same shape: {val.shape} vs {ref.shape}"
    s_val = np.argsort(val)
    s_ref = np.argsort(ref)
    for i in range(val.shape[0]):
        assert test_helpers.dallclose(
            val[i, s_val[i, :]], ref[i, s_ref[i, :]], **kwargs
        ), f"assertion failed for row {i}"


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_rot(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_rot()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_ROT)
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    assert field.shape == (grid.num_vertices, V2E_SIZE)
    assert test_helpers.dallclose(
        field_ref.asnumpy()[horizontal_start:, :], field.asnumpy()[horizontal_start:, :], rtol=rtol
    )


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_geofac_n2s(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.geofac_n2s()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_N2S)
    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_get_geofac_grg(interpolation_savepoint, grid_file, experiment, backend):
    field_ref = interpolation_savepoint.geofac_grg()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field_x = factory.get(attrs.GEOFAC_GRG_X)
    assert field_x.shape == (grid.num_cells, 4)
    field_y = factory.get(attrs.GEOFAC_GRG_Y)
    assert field_y.shape == (grid.num_cells, 4)
    # TODO (@halungge) tolerances are high, especially in the 0th (central) component, check stencil
    #   this passes due to the atol which is too large for the values
    assert test_helpers.dallclose(
        field_ref[0].asnumpy(),
        field_x.asnumpy(),
        rtol=1e-7,
        atol=1e-6,
    )
    assert test_helpers.dallclose(
        field_ref[1].asnumpy(),
        field_y.asnumpy(),
        rtol=1e-7,
        atol=1e-6,
    )


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_get_mass_conserving_cell_average_weight(
    interpolation_savepoint, grid_file, experiment, backend, rtol
):
    field_ref = interpolation_savepoint.c_bln_avg()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.C_BLN_AVG)

    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


## FIXME: does not validate
#   -> connectivity order between reference from serialbox and computed value is different
## TODO (@halungge) rtol is from parametrization is overwritten in assert - function is most probably wrong
#  TODO (@halungge) global grid is not tested
@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
    ],
)
@pytest.mark.datatest
def test_e_flx_avg(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.e_flx_avg()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.E_FLX_AVG)
    assert field.shape == (grid.num_edges, grid.get_connectivity(dims.E2C2EO).shape[1])
    # FIXME: e2c2e constructed from grid file has different ordering than the serialized one
    assert_reordered(field.asnumpy(), field_ref.asnumpy(), rtol=5e-2)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_e_bln_c_s(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.e_bln_c_s()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.E_BLN_C_S)
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_pos_on_tplane_e_x_y(
    interpolation_savepoint: sb.InterpolationSavepoint, grid_file, experiment, backend, rtol
):
    field_ref_1 = interpolation_savepoint.pos_on_tplane_e_x()
    field_ref_2 = interpolation_savepoint.pos_on_tplane_e_y()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    field_1 = factory.get(attrs.POS_ON_TPLANE_E_X)
    field_2 = factory.get(attrs.POS_ON_TPLANE_E_Y)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy(), rtol=rtol)
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), atol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_cells_aw_verts(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.c_intp()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field = factory.get(attrs.CELL_AW_VERTS)

    assert field.shape == (grid.num_vertices, 6)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, rtol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 5e-9),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 1e-11),
    ],
)
@pytest.mark.datatest
def test_nudgecoeffs(interpolation_savepoint, grid_file, experiment, backend, rtol):
    field_ref = interpolation_savepoint.nudgecoeff_e()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    field = factory.get(attrs.NUDGECOEFFS_E)

    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=rtol)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-9),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 3e-2),
    ],
)
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_cell(
    interpolation_savepoint, grid_file, experiment, backend, atol
):
    field_ref_c1 = interpolation_savepoint.rbf_vec_coeff_c1()
    field_ref_c2 = interpolation_savepoint.rbf_vec_coeff_c2()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field_c1 = factory.get(attrs.RBF_VEC_COEFF_C1)
    field_c2 = factory.get(attrs.RBF_VEC_COEFF_C2)
    horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_c1.shape == (grid.num_cells, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
    assert field_c2.shape == (grid.num_cells, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
    assert test_helpers.dallclose(
        field_ref_c1.asnumpy()[horizontal_start:], field_c1.asnumpy()[horizontal_start:], atol=atol
    )
    assert test_helpers.dallclose(
        field_ref_c2.asnumpy()[horizontal_start:], field_c2.asnumpy()[horizontal_start:], atol=atol
    )


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 8e-14),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 2e-9),
    ],
)
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_edge(
    interpolation_savepoint, grid_file, experiment, backend, atol
):
    field_ref_e = interpolation_savepoint.rbf_vec_coeff_e()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field_e = factory.get(attrs.RBF_VEC_COEFF_E)
    horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_e.shape == (grid.num_edges, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE])
    # FIXME: e2c2e constructed from grid file has different ordering than the serialized one
    assert_reordered(
        field_ref_e.asnumpy()[horizontal_start:], field_e.asnumpy()[horizontal_start:], atol=atol
    )


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-10),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 3e-3),
    ],
)
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_vertex(
    interpolation_savepoint, grid_file, experiment, backend, atol
):
    field_ref_v1 = interpolation_savepoint.rbf_vec_coeff_v1()
    field_ref_v2 = interpolation_savepoint.rbf_vec_coeff_v2()
    factory = _get_interpolation_factory(backend, experiment, grid_file)
    grid = factory.grid
    field_v1 = factory.get(attrs.RBF_VEC_COEFF_V1)
    field_v2 = factory.get(attrs.RBF_VEC_COEFF_V2)
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_v1.shape == (grid.num_vertices, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX])
    assert field_v2.shape == (grid.num_vertices, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX])
    assert test_helpers.dallclose(
        field_ref_v1.asnumpy()[horizontal_start:], field_v1.asnumpy()[horizontal_start:], atol=atol
    )
    assert test_helpers.dallclose(
        field_ref_v2.asnumpy()[horizontal_start:], field_v2.asnumpy()[horizontal_start:], atol=atol
    )
