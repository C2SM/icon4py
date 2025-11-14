# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    parallel_geometry_grid,
    parallel_interpolation,
    processor_props,
    ranked_data_path,
)
from ..unit_tests.test_interpolation_factory import assert_reordered


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb

vertex_domain = h_grid.domain(dims.VertexDim)
vert_lb_domain = vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)

@pytest.mark.level("integration")
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name, atol, ind",
    [
        (attrs.C_LIN_E, "c_lin_e", 1e-12, None),
        (attrs.NUDGECOEFFS_E, "nudgecoeff_e", 1e-12, None),
        # (attrs.C_BLN_AVG, "c_bln_avg", 1e-4, None), # fails in EXCLAIM_APE for atol 1e-3
        (attrs.E_BLN_C_S, "e_bln_c_s", 1e-12, None),
        (attrs.GEOFAC_DIV, "geofac_div", 1e-12, None),
        (attrs.GEOFAC_N2S, "geofac_n2s", 1e-12, None),
        (attrs.GEOFAC_GRG_X, "geofac_grg", 1e-3, 0),
        (attrs.GEOFAC_GRG_Y, "geofac_grg", 1e-3, 1),
        (attrs.POS_ON_TPLANE_E_X, "pos_on_tplane_e_x", 1e-12, None),
        (attrs.POS_ON_TPLANE_E_Y, "pos_on_tplane_e_y", 1e-8, None),
        (attrs.CELL_AW_VERTS, "c_intp", 1e-12, None),
    ],
)
def test_distributed_interpolation_attrs(
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    parallel_interpolation: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
    atol: int,
    ind: int | None,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    intp_factory = parallel_interpolation
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)()
    field_ref = field_ref.asnumpy() if ind is None else field_ref[ind].asnumpy()
    field = intp_factory.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=5e-9, atol=atol), f"comparison of {attrs_name} failed"


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name, lb_domain",
    [
        (attrs.GEOFAC_ROT, "geofac_rot", vert_lb_domain),
        # (attrs.E_FLX_AVG, "e_flx_avg", 0), # fails in MCH_CH_R04B09 for atol 1e-2 for atol 1e-2
    ],
)
def test_distributed_interpolation_attrs_reordered(
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    parallel_interpolation: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
    lb_domain: typing.Any,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = parallel_interpolation
    lb = lb_domain if isinstance(lb_domain, int) else factory.grid.start_index(lb_domain)
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)().asnumpy()
    field = factory.get(attrs_name).asnumpy()
    assert_reordered(field[lb:, :], field_ref[lb:, :], atol=2e-9, rtol=1e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name, dim, atol",
    [
        (attrs.RBF_VEC_COEFF_C1, "rbf_vec_coeff_c1", dims.CellDim, 3e-2),
        (attrs.RBF_VEC_COEFF_C2, "rbf_vec_coeff_c2", dims.CellDim, 3e-2),
        (attrs.RBF_VEC_COEFF_E, "rbf_vec_coeff_e", dims.EdgeDim, 7e-1),
        (attrs.RBF_VEC_COEFF_V1, "rbf_vec_coeff_v1", dims.VertexDim, 3e-3),
        (attrs.RBF_VEC_COEFF_V2, "rbf_vec_coeff_v2", dims.VertexDim, 3e-3)
    ],
)
def test_distributed_interpolation_rbf(
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    parallel_interpolation: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
    dim: gtx.Dimension,
    atol: int,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = parallel_interpolation
    owner_mask = decomposition_info._owner_mask[dim]
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)().asnumpy()
    field = factory.get(attrs_name).asnumpy()
    assert_reordered(field[owner_mask], field_ref[owner_mask], atol=atol)
