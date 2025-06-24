# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers


@pytest.mark.parametrize(
    "experiment, model_top_height, damping_height, stretch_factor",
    [
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, 30000.0, 8000.0, 0.85),
    ],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
@pytest.mark.parametrize("location", ["nwp-gscp-interface", "interface-nwp"])
def test_saturation_adjustement(
    experiment,
    location,
    model_top_height,
    damping_height,
    stretch_factor,
    date,
    data_provider,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    backend,
):
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)

    config = satad.SaturationAdjustmentConfig(
        tolerance=1e-3,
        max_iter=10,
    )
    dtime = 2.0

    vertical_config = v_grid.VerticalGridConfig(icon_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    qv_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)
    qc_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)

    metric_state = satad.MetricStateSaturationAdjustment(
        ddqz_z_full=metrics_savepoint.ddqz_z_full()
    )

    saturation_adjustment = satad.SaturationAdjustment(
        config=config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qv = satad_init.qv()
    qc = satad_init.qc()
    rho = satad_init.rho()
    temperature = satad_init.temperature()

    # run saturation adjustment
    saturation_adjustment.run(
        dtime=dtime,
        rho=rho,
        temperature=temperature,
        qv=qv,
        qc=qc,
        temperature_tendency=temperature_tendency,
        qv_tendency=qv_tendency,
        qc_tendency=qc_tendency,
    )

    updated_qv = qv.asnumpy() + qv_tendency.asnumpy() * dtime
    updated_qc = qc.asnumpy() + qc_tendency.asnumpy() * dtime
    updated_temperature = temperature.asnumpy() + temperature_tendency.asnumpy() * dtime

    assert helpers.dallclose(
        updated_qv,
        satad_exit.qv().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_qc,
        satad_exit.qc().asnumpy(),
        atol=1.0e-13,
    )
    assert helpers.dallclose(
        updated_temperature,
        satad_exit.temperature().asnumpy(),
        atol=1.0e-13,
    )
