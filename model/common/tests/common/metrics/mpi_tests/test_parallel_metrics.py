# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.metrics import metrics_attributes as attrs, metrics_factory
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
    "attrs_name, metrics_name",
    [
        (attrs.DDQZ_Z_FULL_E, "ddqz_z_full_e"),
        (attrs.ZDIFF_GRADP, "zdiff_gradp"),
        (attrs.VERTOFFSET_GRADP, "vertoffset_gradp"),
        (attrs.Z_MC, "z_mc"),
        (attrs.DDQZ_Z_HALF, "ddqz_z_half"),
        (attrs.SCALING_FACTOR_FOR_3D_DIVDAMP, "scalfac_dd3d"),
        (attrs.RAYLEIGH_W, "rayleigh_w"),
        (attrs.COEFF_GRADEKIN, "coeff_gradekin"),
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
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = parallel_metrics

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-8, atol=1.0e-8)
