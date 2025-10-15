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

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers, test_utils
from model.common.tests.common.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)
from model.common.tests.common.interpolation.unit_tests.test_interpolation_factory import (
    _get_interpolation_factory,
    assert_reordered,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.testing import serialbox as sb

vertex_domain = h_grid.domain(dims.VertexDim)
vert_lb_domain = vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name",
    [
        ("interpolation_coefficient_from_cell_to_edge", "c_lin_e"),
        ("nudging_coefficients_for_edges", "nudgecoeff_e"),
        ("bilinear_edge_cell_weight", "e_bln_c_s"),
        ("geometrical_factor_for_nabla_2_scalar", "geofac_n2s"),
        ("cell_to_vertex_interpolation_factor_by_area_weighting", "c_intp"),
    ],
)
def test_distributed_interpolation_attrs(
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    attrs_name: str,
    intrp_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    factory = _get_interpolation_factory(backend, experiment)
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)().asnumpy()
    field = factory.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=5e-9)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name, lb_domain",
    [
        ("geometrical_factor_for_curl", "geofac_rot", vert_lb_domain),
        ("rbf_interpolation_coefficient_edge", "rbf_vec_coeff_e", vert_lb_domain),
        ("geometrical_factor_for_gradient_of_divergence", "geofac_grdiv", 0),
    ],
)
def test_distributed_interpolation_attrs_reordered(
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    attrs_name: str,
    intrp_name: str,
    lb_domain: typing.Any,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    factory = _get_interpolation_factory(backend, experiment)
    lb = factory.grid.start_index(lb_domain) if not isinstance(lb_domain, int) else lb_domain
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)().asnumpy()
    field = factory.get(attrs_name).asnumpy()
    assert_reordered(field[lb:, :], field_ref[lb:, :], atol=2e-9, rtol=5e-9)
