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
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils
from model.common.tests.common.interpolation.unit_tests.test_interpolation_factory import (
    _get_interpolation_factory,
    assert_reordered,
)

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
    topography_savepoint,
)
from ..unit_tests.test_metrics_factory import _get_metrics_factory


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb

vertex_domain = h_grid.domain(dims.VertexDim)
vert_lb_domain = vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        ("functional_determinant_of_metrics_on_full_levels_on_edges", "ddqz_z_full_e"),
        # ("ddxt_z_half_e", "ddxt_z_half_e"), DDXT_Z_HALF_E
        ("zdiff_gradp", "zdiff_gradp"),
        ("height", "z_mc"),
        ("functional_determinant_of_metrics_on_interface_levels", "ddqz_z_half"),
        ("scaling_factor_for_3d_divergence_damping", "scalfac_dd3d"),
        ("rayleigh_w", "rayleigh_w"),
        ("coeff_gradekin", "coeff_gradekin"),
        ("mask_prog_halo_c", "mask_prog_halo_c"),
        ("zd_intcoef_dsl", "zd_intcoef"),
    ],
)
def test_distributed_metrics_attrs(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    attrs_name: str,
    metrics_name: str,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
        exchange=exchange,
    )
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    field = factory.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-8, atol=1.0e-10)
