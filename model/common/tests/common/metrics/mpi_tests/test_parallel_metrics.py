# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry, horizontal as h_grid
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    parallel_geometry_grid,
    parallel_interpolation,
    parallel_metrics,
    processor_props,
    ranked_data_path,
    topography_savepoint,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb

vertex_domain = h_grid.domain(dims.VertexDim)
vert_lb_domain = vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)


@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name, k_lb, reshape",
    [
        ("functional_determinant_of_metrics_on_full_levels_on_edges", "ddqz_z_full_e", 0, False),
        ("zdiff_gradp", "zdiff_gradp", 0, True),
        ("height", "z_mc", 0, False),
        ("functional_determinant_of_metrics_on_interface_levels", "ddqz_z_half", 1, False),
        ("scaling_factor_for_3d_divergence_damping", "scalfac_dd3d", 0, False),
        ("rayleigh_w", "rayleigh_w", 0, False),
        ("coeff_gradekin", "coeff_gradekin", 0, False),
    ],
)
@pytest.mark.parametrize("experiment", [test_defs.Experiments.EXCLAIM_APE])
def test_distributed_metrics_attrs(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    parallel_metrics: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
    k_lb: int,
    reshape: bool,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = parallel_metrics

    field = factory.get(attrs_name).asnumpy()
    field = field if k_lb == 0 else field[:, k_lb:]

    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    field_ref = field_ref if k_lb == 0 else field_ref[:, k_lb:]
    field_ref = field_ref.reshape(field_ref.shape[0], -1) if reshape else field_ref
    assert test_utils.dallclose(field, field_ref, rtol=1e-8, atol=1.0e-8)
