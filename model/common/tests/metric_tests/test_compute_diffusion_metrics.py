# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import pytest
from gt4py.next import as_field

from icon4py.model.common.dimension import (
    C2E2CDim,
    CECDim,
    CellDim,
    KDim,
)
from icon4py.model.common.grid.horizontal import (
    HorizontalMarkerIndex,
)
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
    constant_field,
    dallclose,
    flatten_first_two_dims,
    is_roundtrip,
    zero_field,
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_diffusion_metrics(
    metrics_savepoint, experiment, interpolation_savepoint, icon_grid, grid_savepoint, backend
):
    if is_roundtrip(backend):
        pytest.skip("skipping: slow backend")

    if experiment == dt_utils.GLOBAL_EXPERIMENT:
        pytest.skip(f"Fields not computed for {experiment}")

    mask_hdiff = zero_field(icon_grid, CellDim, KDim, dtype=bool).asnumpy()
    zd_vertoffset_dsl = zero_field(icon_grid, CellDim, C2E2CDim, KDim).asnumpy()
    z_vintcoeff = zero_field(icon_grid, CellDim, C2E2CDim, KDim).asnumpy()
    zd_intcoef_dsl = zero_field(icon_grid, CellDim, C2E2CDim, KDim).asnumpy()
    z_maxslp_avg = zero_field(icon_grid, CellDim, KDim)
    z_maxhgtd_avg = zero_field(icon_grid, CellDim, KDim)
    zd_diffcoef_dsl = zero_field(icon_grid, CellDim, KDim).asnumpy()
    maxslp = zero_field(icon_grid, CellDim, KDim)
    maxhgtd = zero_field(icon_grid, CellDim, KDim)
    max_nbhgt = zero_field(icon_grid, CellDim)

    c2e2c = icon_grid.connectivities[C2E2CDim]
    nbidx = constant_field(icon_grid, 1, CellDim, C2E2CDim, KDim, dtype=int).asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg()
    thslp_zdiffu = 0.02
    thhgtd_zdiffu = 125
    cell_nudging = icon_grid.get_start_index(CellDim, HorizontalMarkerIndex.nudging(CellDim))
    cell_lateral = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    nlev = icon_grid.num_levels

    compute_maxslp_maxhgtd.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        z_maxslp=maxslp,
        z_maxhgtd=maxhgtd,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"C2E": icon_grid.get_offset_provider("C2E")},
    )

    z_mc = zero_field(icon_grid, CellDim, KDim, extend={KDim: 1})
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
        z_maxslp_avg=z_maxslp_avg,
        z_maxhgtd_avg=z_maxhgtd_avg,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={
            "C2E2C": icon_grid.get_offset_provider("C2E2C"),
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
        },
    )

    compute_max_nbhgt.with_backend(backend)(
        z_mc_nlev=as_field((CellDim,), z_mc.asnumpy()[:, nlev - 1]),
        max_nbhgt=max_nbhgt,
        horizontal_start=cell_nudging,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E2C": icon_grid.get_offset_provider("C2E2C")},
    )

    mask_hdiff, zd_diffcoef_dsl, zd_intcoef_dsl, zd_vertoffset_dsl = compute_diffusion_metrics(
        z_mc=z_mc.asnumpy(),
        z_mc_off=z_mc.asnumpy()[c2e2c],
        max_nbhgt=max_nbhgt.asnumpy(),
        c_owner_mask=grid_savepoint.c_owner_mask().asnumpy(),
        nbidx=nbidx,
        z_vintcoeff=z_vintcoeff,
        z_maxslp_avg=z_maxslp_avg.asnumpy(),
        z_maxhgtd_avg=z_maxhgtd_avg.asnumpy(),
        mask_hdiff=mask_hdiff,
        zd_diffcoef_dsl=zd_diffcoef_dsl,
        zd_intcoef_dsl=zd_intcoef_dsl,
        zd_vertoffset_dsl=zd_vertoffset_dsl,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        cell_nudging=cell_nudging,
        n_cells=icon_grid.num_cells,
        nlev=nlev,
    )
    zd_intcoef_dsl = flatten_first_two_dims(
        CECDim, KDim, field=as_field((CellDim, C2E2CDim, KDim), zd_intcoef_dsl)
    )
    zd_vertoffset_dsl = flatten_first_two_dims(
        CECDim, KDim, field=as_field((CellDim, C2E2CDim, KDim), zd_vertoffset_dsl)
    )

    assert dallclose(mask_hdiff, metrics_savepoint.mask_hdiff().asnumpy())
    assert dallclose(zd_diffcoef_dsl, metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11)
    assert dallclose(zd_vertoffset_dsl.asnumpy(), metrics_savepoint.zd_vertoffset().asnumpy())
    assert dallclose(zd_intcoef_dsl.asnumpy(), metrics_savepoint.zd_intcoef().asnumpy())
