# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

import icon4py.model.common.type_alias as ta
import icon4py.model.testing.helpers as helpers
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.metrics.reference_atmosphere import (
    compute_d2dexdz2_fac_mc,
    compute_d_exner_dz_ref_ic,
    compute_reference_atmosphere_cell_fields,
    compute_reference_atmosphere_edge_fields,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_reference_atmosphere_fields_on_full_level_masspoints(
    icon_grid, metrics_savepoint, experiment, backend
):
    exner_ref_mc_ref = metrics_savepoint.exner_ref_mc()
    rho_ref_mc_ref = metrics_savepoint.rho_ref_mc()
    theta_ref_mc_ref = metrics_savepoint.theta_ref_mc()

    z_mc = metrics_savepoint.z_mc()

    exner_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    rho_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    theta_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    compute_reference_atmosphere_cell_fields.with_backend(backend)(
        z_height=z_mc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_mc,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert helpers.dallclose(rho_ref_mc.asnumpy(), rho_ref_mc_ref.asnumpy())
    assert helpers.dallclose(theta_ref_mc.asnumpy(), theta_ref_mc_ref.asnumpy())
    assert helpers.dallclose(exner_ref_mc.asnumpy(), exner_ref_mc_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_reference_atmosphere_on_half_level_mass_points(
    icon_grid, metrics_savepoint, experiment, backend
):
    theta_ref_ic_ref = metrics_savepoint.theta_ref_ic()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat, backend=backend
    )
    rho_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat, backend=backend
    )
    theta_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat, backend=backend
    )
    compute_reference_atmosphere_cell_fields.with_backend(backend=backend)(
        z_height=z_ifc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_ic,
        rho_ref_mc=rho_ref_ic,
        theta_ref_mc=theta_ref_ic,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={},
    )

    assert helpers.dallclose(theta_ref_ic.asnumpy(), theta_ref_ic_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_d_exner_dz_ref_ic(icon_grid, metrics_savepoint, experiment, backend):
    theta_ref_ic = metrics_savepoint.theta_ref_ic()
    d_exner_dz_ref_ic_ref = metrics_savepoint.d_exner_dz_ref_ic()
    d_exner_dz_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
    )
    compute_d_exner_dz_ref_ic.with_backend(backend)(
        theta_ref_ic=theta_ref_ic,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        out=d_exner_dz_ref_ic,
        offset_provider={},
    )

    assert helpers.dallclose(d_exner_dz_ref_ic.asnumpy(), d_exner_dz_ref_ic_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_reference_atmosphere_on_full_level_edge_fields(
    icon_grid, interpolation_savepoint, metrics_savepoint, experiment, backend
):
    rho_ref_me_ref = metrics_savepoint.rho_ref_me()
    theta_ref_me_ref = metrics_savepoint.theta_ref_me()
    rho_ref_me = metrics_savepoint.rho_ref_me()
    theta_ref_me = metrics_savepoint.theta_ref_me()
    c_lin_e = interpolation_savepoint.c_lin_e()

    z_mc = metrics_savepoint.z_mc()
    z_me = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, backend=backend
    )
    horizontal_start = icon_grid.start_index(
        horizontal.domain(dims.EdgeDim)(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    cell_2_edge_interpolation.with_backend(backend)(
        z_mc,
        c_lin_e,
        z_me,
        horizontal_start=horizontal_start,
        horizontal_end=gtx.int32(icon_grid.num_edges),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={"E2C": icon_grid.get_connectivity("E2C")},
    )
    compute_reference_atmosphere_edge_fields.with_backend(backend)(
        z_me=z_me,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        del_t_bg=constants.DELTA_TEMPERATURE,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        horizontal_start=horizontal_start,
        horizontal_end=(gtx.int32(icon_grid.num_edges)),
        vertical_start=(gtx.int32(0)),
        vertical_end=(gtx.int32(icon_grid.num_levels)),
        offset_provider={},
    )
    assert helpers.dallclose(rho_ref_me.asnumpy(), rho_ref_me_ref.asnumpy(), rtol=1e-10)
    assert helpers.dallclose(theta_ref_me.asnumpy(), theta_ref_me_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_d2dexdz2_fac_mc(icon_grid, metrics_savepoint, grid_savepoint, experiment, backend):
    z_mc = metrics_savepoint.z_mc()
    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()

    d2dexdz2_fac1_mc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)
    d2dexdz2_fac2_mc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend)

    compute_d2dexdz2_fac_mc.with_backend(backend=backend)(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        exner_ref_mc=metrics_savepoint.exner_ref_mc(),
        z_mc=z_mc,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        cpd=constants.CPD,
        grav=constants.GRAV,
        del_t_bg=constants.DEL_T_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert helpers.dallclose(d2dexdz2_fac1_mc.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert helpers.dallclose(d2dexdz2_fac2_mc.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())
