# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
import icon4py.model.common.type_alias as ta
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.metrics.metric_fields import compute_z_mc
from icon4py.model.common.metrics.reference_atmosphere import (
    compute_d_exner_dz_ref_ic,
    compute_reference_atmosphere_cell_fields,
    compute_reference_atmosphere_edge_fields,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils


# TODO (@halungge) some tests need to run on a compiled backend: embedded does not work with the
#  Koff[-1] and roundtrip is too slow on the large grid


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_reference_atmosphere_fields_on_full_level_masspoints(
    icon_grid, metrics_savepoint, backend
):
    if helpers.is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    exner_ref_mc_ref = metrics_savepoint.exner_ref_mc()
    rho_ref_mc_ref = metrics_savepoint.rho_ref_mc()
    theta_ref_mc_ref = metrics_savepoint.theta_ref_mc()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_mc = helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
    rho_ref_mc = helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
    theta_ref_mc = helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
    z_mc = helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
    start = 0
    horizontal_end = icon_grid.num_cells
    vertical_end = icon_grid.num_levels
    compute_z_mc.with_backend(backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    compute_reference_atmosphere_cell_fields.with_backend(backend)(
        z_height=z_mc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEAL_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_mc,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={},
    )

    assert helpers.dallclose(rho_ref_mc.asnumpy(), rho_ref_mc_ref.asnumpy())
    assert helpers.dallclose(theta_ref_mc.asnumpy(), theta_ref_mc_ref.asnumpy())
    assert helpers.dallclose(exner_ref_mc.asnumpy(), exner_ref_mc_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_reference_atmsophere_on_half_level_mass_points(
    icon_grid, metrics_savepoint, backend
):
    if helpers.is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    theta_ref_ic_ref = metrics_savepoint.theta_ref_ic()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_ic = helpers.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
    )
    rho_ref_ic = helpers.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
    )
    theta_ref_ic = helpers.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
    )
    start = 0
    horizontal_end = icon_grid.num_cells
    vertical_end = icon_grid.num_levels + 1

    compute_reference_atmosphere_cell_fields.with_backend(backend=backend)(
        z_height=z_ifc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEAL_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_ic,
        rho_ref_mc=rho_ref_ic,
        theta_ref_mc=theta_ref_ic,
        horizontal_start=start,
        horizontal_end=horizontal_end,
        vertical_start=start,
        vertical_end=vertical_end,
        offset_provider={},
    )

    assert helpers.dallclose(theta_ref_ic.asnumpy(), theta_ref_ic_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_d_exner_dz_ref_ic(icon_grid, metrics_savepoint, backend):
    if helpers.is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    theta_ref_ic = metrics_savepoint.theta_ref_ic()
    d_exner_dz_ref_ic_ref = metrics_savepoint.d_exner_dz_ref_ic()
    d_exner_dz_ref_ic = helpers.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
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
    icon_grid, interpolation_savepoint, metrics_savepoint, backend
):
    if helpers.is_roundtrip(backend):
        pytest.skip("skipping: slow backend")
    rho_ref_me_ref = metrics_savepoint.rho_ref_me()
    theta_ref_me_ref = metrics_savepoint.theta_ref_me()
    rho_ref_me = metrics_savepoint.rho_ref_me()
    theta_ref_me = metrics_savepoint.theta_ref_me()
    c_lin_e = interpolation_savepoint.c_lin_e()

    z_ifc = metrics_savepoint.z_ifc()
    z_mc = helpers.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
    z_me = helpers.zero_field(icon_grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
    horizontal_start = icon_grid.start_index(
        horizontal.domain(dims.EdgeDim)(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    num_cells = gtx.int32(icon_grid.num_cells)
    num_edges = int(icon_grid.num_edges)
    vertical_start = 0
    vertical_end = gtx.int32(icon_grid.num_levels)
    compute_z_mc.with_backend(backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        horizontal_start=0,
        horizontal_end=num_cells,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )
    cell_2_edge_interpolation.with_backend(backend)(
        z_mc,
        c_lin_e,
        z_me,
        horizontal_start=horizontal_start,
        horizontal_end=num_edges,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    compute_reference_atmosphere_edge_fields.with_backend(backend)(
        z_me=z_me,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEAL_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        horizontal_start=horizontal_start,
        horizontal_end=num_edges,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={},
    )
    assert helpers.dallclose(rho_ref_me.asnumpy(), rho_ref_me_ref.asnumpy())
    assert helpers.dallclose(theta_ref_me.asnumpy(), theta_ref_me_ref.asnumpy())
