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
    geometry_from_savepoint,
    grid_savepoint,
    icon_grid,
    interpolation_factory_from_savepoint,
    metrics_factory_from_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
    topography_savepoint,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.CELL_HEIGHT_ON_HALF_LEVEL, "z_ifc"),
        (attrs.DDQZ_Z_FULL_E, "ddqz_z_full_e"),
        # (attrs.ZDIFF_GRADP, "zdiff_gradp"),
        # (attrs.VERTOFFSET_GRADP, "vertoffset_gradp"), #atol=1.0e-5
        (attrs.Z_MC, "z_mc"),
        (attrs.DDQZ_Z_HALF, "ddqz_z_half"),
        (attrs.SCALING_FACTOR_FOR_3D_DIVDAMP, "scalfac_dd3d"),
        (attrs.RAYLEIGH_W, "rayleigh_w"),
        # (attrs.COEFF_GRADEKIN, "coeff_gradekin"), # check: possibly bounds?
    ],
)
def test_distributed_metrics_attrs(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint
    print(f"computed flatlev {factory.vertical_grid.nflatlev}, expected{grid_savepoint.nflatlev()}")

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-8, atol=1.0e-8)
