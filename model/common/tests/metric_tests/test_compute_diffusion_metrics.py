# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_diffusion_metrics import (
    compute_diffusion_metrics,
)
from icon4py.model.common.metrics.metric_fields import (
    compute_max_nbhgt,
    compute_maxslp_maxhgtd,
    compute_weighted_cell_neighbor_sum,
    compute_z_mc,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.helpers import (
    dallclose,
    is_roundtrip,
    zero_field,
)


# TODO (halungge) fails in embedded
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_diffusion_metrics(
    metrics_savepoint, experiment, interpolation_savepoint, icon_grid, grid_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")

    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        pytest.skip(f"Fields not computed for {experiment}")

    maxslp_avg = zero_field(icon_grid, dims.CellDim, dims.KDim)
    maxhgtd_avg = zero_field(icon_grid, dims.CellDim, dims.KDim)
    maxslp = zero_field(icon_grid, dims.CellDim, dims.KDim)
    maxhgtd = zero_field(icon_grid, dims.CellDim, dims.KDim)
    max_nbhgt = zero_field(icon_grid, dims.CellDim)

    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    c_bln_avg = interpolation_savepoint.c_bln_avg()
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125.0
    cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))

    cell_lateral = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    nlev = icon_grid.num_levels

    compute_maxslp_maxhgtd.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    z_mc = zero_field(icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
    compute_z_mc.with_backend(backend)(
        metrics_savepoint.z_ifc(),
        z_mc,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    compute_weighted_cell_neighbor_sum.with_backend(backend)(
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        c_bln_avg=c_bln_avg,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
        },
    )

    compute_max_nbhgt.with_backend(backend)(
        z_mc_nlev=gtx.as_field((dims.CellDim,), z_mc.asnumpy()[:, nlev - 1]),
        max_nbhgt=max_nbhgt,
        horizontal_start=cell_nudging,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E2C": icon_grid.get_offset_provider("C2E2C")},
    )

    mask_hdiff, zd_diffcoef_dsl, zd_intcoef_dsl, zd_vertoffset_dsl = compute_diffusion_metrics(
        c2e2c=c2e2c,
        z_mc=z_mc.asnumpy(),
        max_nbhgt=max_nbhgt.asnumpy(),
        c_owner_mask=grid_savepoint.c_owner_mask().asnumpy(),
        maxslp_avg=maxslp_avg.asnumpy(),
        maxhgtd_avg=maxhgtd_avg.asnumpy(),
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        n_c2e2c=c2e2c.shape[1],
        cell_nudging=cell_nudging,
        n_cells=icon_grid.num_cells,
        nlev=nlev,
    )

    assert dallclose(mask_hdiff, metrics_savepoint.mask_hdiff().asnumpy())
    assert dallclose(zd_diffcoef_dsl, metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11)
    assert dallclose(zd_vertoffset_dsl, metrics_savepoint.zd_vertoffset().asnumpy())
    assert dallclose(zd_intcoef_dsl, metrics_savepoint.zd_intcoef().asnumpy())
