# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.common import dimension as dims, exceptions
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.io import cf_utils
from icon4py.model.common.metrics import metric_fields as mf
from icon4py.model.common.metrics.compute_wgtfacq import (
    compute_wgtfacq_c_dsl,
    compute_wgtfacq_e_dsl,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import factory


@pytest.mark.datatest
def test_factory_check_dependencies_on_register(icon_grid, backend):
    fields_factory = factory.FieldsFactory(icon_grid, backend)
    provider = factory.ProgramFieldProvider(
        func=mf.compute_z_mc,
        domain={dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (0, icon_grid.num_levels)},
        fields={"z_mc": "height"},
        deps={"z_ifc": "height_on_interface_levels"},
    )
    with pytest.raises(ValueError) as e:
        fields_factory.register_provider(provider)
        assert e.value.match("'height_on_interface_levels' not found")


@pytest.mark.datatest
def test_factory_raise_error_if_no_grid_is_set(metrics_savepoint):
    z_ifc = metrics_savepoint.z_ifc()
    k_index = gtx.as_field((dims.KDim,), xp.arange(1, dtype=gtx.int32))
    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {"height_on_interface_levels": z_ifc, cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index}
    )
    fields_factory = factory.FieldsFactory(grid=None)
    fields_factory.register_provider(pre_computed_fields)
    with pytest.raises(exceptions.IncompleteSetupError) as e:
        fields_factory.get("height_on_interface_levels")
        assert e.value.match("not fully instantiated")


@pytest.mark.datatest
def test_factory_returns_field(metrics_savepoint, icon_grid, backend):
    z_ifc = metrics_savepoint.z_ifc()
    k_index = gtx.as_field((dims.KDim,), xp.arange(icon_grid.num_levels + 1, dtype=gtx.int32))
    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {"height_on_interface_levels": z_ifc, cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index}
    )
    fields_factory = factory.FieldsFactory()
    fields_factory.register_provider(pre_computed_fields)
    fields_factory.with_grid(icon_grid).with_allocator(backend)
    field = fields_factory.get("height_on_interface_levels", factory.RetrievalType.FIELD)
    assert field.ndarray.shape == (icon_grid.num_cells, icon_grid.num_levels + 1)
    meta = fields_factory.get("height_on_interface_levels", factory.RetrievalType.METADATA)
    assert meta["standard_name"] == "height_on_interface_levels"
    assert meta["dims"] == (
        dims.CellDim,
        dims.KHalfDim,
    )
    assert meta["units"] == "m"
    data_array = fields_factory.get("height_on_interface_levels", factory.RetrievalType.DATA_ARRAY)
    assert data_array.data.shape == (icon_grid.num_cells, icon_grid.num_levels + 1)
    assert data_array.data.dtype == xp.float64
    for key in ("dims", "standard_name", "units", "icon_var_name"):
        assert key in data_array.attrs.keys()


@pytest.mark.datatest
def test_field_provider_for_program(icon_grid, metrics_savepoint, backend):
    fields_factory = factory.FieldsFactory(icon_grid, backend)
    k_index = gtx.as_field((dims.KDim,), xp.arange(icon_grid.num_levels + 1, dtype=gtx.int32))
    z_ifc = metrics_savepoint.z_ifc()

    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {"height_on_interface_levels": z_ifc, cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index}
    )

    fields_factory.register_provider(pre_computed_fields)

    height_provider = factory.ProgramFieldProvider(
        func=mf.compute_z_mc,
        domain={
            dims.CellDim: (
                HorizontalMarkerIndex.local(dims.CellDim),
                HorizontalMarkerIndex.end(dims.CellDim),
            ),
            dims.KDim: (0, icon_grid.num_levels),
        },
        fields={"z_mc": "height"},
        deps={"z_ifc": "height_on_interface_levels"},
    )
    fields_factory.register_provider(height_provider)
    functional_determinant_provider = factory.ProgramFieldProvider(
        func=mf.compute_ddqz_z_half,
        domain={
            dims.CellDim: (
                HorizontalMarkerIndex.local(dims.CellDim),
                HorizontalMarkerIndex.end(dims.CellDim),
            ),
            dims.KHalfDim: (0, icon_grid.num_levels + 1),
        },
        fields={"ddqz_z_half": "functional_determinant_of_metrics_on_interface_levels"},
        deps={
            "z_ifc": "height_on_interface_levels",
            "z_mc": "height",
            "k": cf_utils.INTERFACE_LEVEL_STANDARD_NAME,
        },
        params={"nlev": icon_grid.num_levels},
    )
    fields_factory.register_provider(functional_determinant_provider)
    data = fields_factory.get(
        "functional_determinant_of_metrics_on_interface_levels", type_=factory.RetrievalType.FIELD
    )
    ref = metrics_savepoint.ddqz_z_half().ndarray
    assert helpers.dallclose(data.ndarray, ref)


def test_field_provider_for_numpy_function(
    icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    fields_factory = factory.FieldsFactory(grid=icon_grid, backend=backend)
    k_index = gtx.as_field((dims.KDim,), xp.arange(icon_grid.num_levels + 1, dtype=gtx.int32))
    z_ifc = metrics_savepoint.z_ifc()
    wgtfacq_c_ref = metrics_savepoint.wgtfacq_c_dsl()

    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {"height_on_interface_levels": z_ifc, cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index}
    )
    fields_factory.register_provider(pre_computed_fields)
    func = compute_wgtfacq_c_dsl
    deps = {"z_ifc": "height_on_interface_levels"}
    params = {"nlev": icon_grid.num_levels}
    compute_wgtfacq_c_provider = factory.NumpyFieldsProvider(
        func=func,
        domain={
            dims.CellDim: (0, HorizontalMarkerIndex.end(dims.CellDim)),
            dims.KDim: (0, icon_grid.num_levels),
        },
        fields=["weighting_factor_for_quadratic_interpolation_to_cell_surface"],
        deps=deps,
        params=params,
    )
    fields_factory.register_provider(compute_wgtfacq_c_provider)

    wgtfacq_c = fields_factory.get(
        "weighting_factor_for_quadratic_interpolation_to_cell_surface", factory.RetrievalType.FIELD
    )

    assert helpers.dallclose(wgtfacq_c.asnumpy(), wgtfacq_c_ref.asnumpy())


def test_field_provider_for_numpy_function_with_offsets(
    icon_grid, metrics_savepoint, interpolation_savepoint, backend
):
    fields_factory = factory.FieldsFactory(grid=icon_grid, backend=backend)
    k_index = gtx.as_field((dims.KDim,), xp.arange(icon_grid.num_levels + 1, dtype=gtx.int32))
    z_ifc = metrics_savepoint.z_ifc()
    c_lin_e = interpolation_savepoint.c_lin_e()
    wgtfacq_e_ref = metrics_savepoint.wgtfacq_e_dsl(icon_grid.num_levels + 1)

    pre_computed_fields = factory.PrecomputedFieldsProvider(
        {
            "height_on_interface_levels": z_ifc,
            cf_utils.INTERFACE_LEVEL_STANDARD_NAME: k_index,
            "c_lin_e": c_lin_e,
        }
    )
    fields_factory.register_provider(pre_computed_fields)
    func = compute_wgtfacq_c_dsl
    params = {"nlev": icon_grid.num_levels}
    compute_wgtfacq_c_provider = factory.NumpyFieldsProvider(
        func=func,
        domain={dims.CellDim: (0, icon_grid.num_cells), dims.KDim: (0, icon_grid.num_levels)},
        fields=["weighting_factor_for_quadratic_interpolation_to_cell_surface"],
        deps={"z_ifc": "height_on_interface_levels"},
        params=params,
    )
    deps = {
        "z_ifc": "height_on_interface_levels",
        "wgtfacq_c_dsl": "weighting_factor_for_quadratic_interpolation_to_cell_surface",
        "c_lin_e": "c_lin_e",
    }
    fields_factory.register_provider(compute_wgtfacq_c_provider)
    wgtfacq_e_provider = factory.NumpyFieldsProvider(
        func=compute_wgtfacq_e_dsl,
        deps=deps,
        offsets={"e2c": dims.E2CDim},
        domain={dims.EdgeDim: (0, icon_grid.num_edges), dims.KDim: (0, icon_grid.num_levels)},
        fields=["weighting_factor_for_quadratic_interpolation_to_edge_center"],
        params={"n_edges": icon_grid.num_edges, "nlev": icon_grid.num_levels},
    )

    fields_factory.register_provider(wgtfacq_e_provider)
    wgtfacq_e = fields_factory.get(
        "weighting_factor_for_quadratic_interpolation_to_edge_center", factory.RetrievalType.FIELD
    )

    assert helpers.dallclose(wgtfacq_e.asnumpy(), wgtfacq_e_ref.asnumpy())
