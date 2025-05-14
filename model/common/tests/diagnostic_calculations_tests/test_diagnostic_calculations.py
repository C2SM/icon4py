# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.diagnostic_calculations.stencils import (
    diagnose_pressure,
    diagnose_surface_pressure,
    diagnose_temperature,
)
from icon4py.model.common.interpolation.stencils import (
    edge_2_cell_vector_rbf_interpolation as rbf,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, helpers


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_temperature(
    experiment,
    data_provider,
    icon_grid,
    backend,
):
    diagnostic_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    temperature_ref = diagnostic_reference_savepoint.temperature().asnumpy()
    virtual_temperature_ref = diagnostic_reference_savepoint.virtual_temperature().asnumpy()
    initial_prognostic_savepoint = data_provider.from_savepoint_prognostics_initial()
    exner = initial_prognostic_savepoint.exner_now()
    theta_v = initial_prognostic_savepoint.theta_v_now()

    temperature = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
    )
    virtual_temperature = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
    )

    qv = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    qc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    qr = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    qi = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    qs = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    qg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)

    diagnose_temperature.diagnose_virtual_temperature_and_temperature.with_backend(backend)(
        qv=qv,
        qc=qc,
        qr=qr,
        qi=qi,
        qs=qs,
        qg=qg,
        theta_v=theta_v,
        exner=exner,
        virtual_temperature=virtual_temperature,
        temperature=temperature,
        rv_o_rd_minus1=phy_const.RV_O_RD_MINUS_1,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    # only temperature is tested because there is no moisture in the JW test. i.e. temperature = virtual_temperature
    assert helpers.dallclose(
        temperature.asnumpy(),
        temperature_ref,
    )
    assert helpers.dallclose(
        virtual_temperature.asnumpy(),
        virtual_temperature_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_meridional_and_zonal_winds(
    experiment,
    data_provider,
    interpolation_savepoint,
    icon_grid,
    backend,
):
    prognostics_init_savepoint = data_provider.from_savepoint_prognostics_initial()
    vn = prognostics_init_savepoint.vn_now()
    rbv_vec_coeff_c1 = interpolation_savepoint.rbf_vec_coeff_c1()
    rbv_vec_coeff_c2 = interpolation_savepoint.rbf_vec_coeff_c2()

    diagnostics_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    u_ref = diagnostics_reference_savepoint.zonal_wind().asnumpy()
    v_ref = diagnostics_reference_savepoint.meridional_wind().asnumpy()

    u = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)
    v = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend)

    cell_domain = h_grid.domain(dims.CellDim)
    cell_end_lateral_boundary_level_2 = icon_grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.END))

    rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=vn,
        ptr_coeff_1=rbv_vec_coeff_c1,
        ptr_coeff_2=rbv_vec_coeff_c2,
        p_u_out=u,
        p_v_out=v,
        horizontal_start=cell_end_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "C2E2C2E": icon_grid.get_offset_provider("C2E2C2E"),
        },
    )

    assert helpers.dallclose(
        u.asnumpy(),
        u_ref,
    )

    assert helpers.dallclose(
        v.asnumpy(),
        v_ref,
        atol=1.0e-13,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_surface_pressure(
    experiment, data_provider, icon_grid, backend, metrics_savepoint
):
    initial_diagnostic_savepoint = data_provider.from_savepoint_diagnostics_initial()
    surface_pressure_ref = initial_diagnostic_savepoint.pressure_sfc().asnumpy()
    initial_prognostic_savepoint = data_provider.from_savepoint_prognostics_initial()
    exner = initial_prognostic_savepoint.exner_now()
    virtual_temperature = initial_diagnostic_savepoint.virtual_temperature()
    ddqz_z_full = metrics_savepoint.ddqz_z_full()

    surface_pressure = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, backend=backend
    )

    cell_domain = h_grid.domain(dims.CellDim)

    diagnose_surface_pressure.diagnose_surface_pressure.with_backend(backend)(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=surface_pressure,
        cpd_o_rd=phy_const.CPD_O_RD,
        p0ref=phy_const.P0REF,
        grav_o_rd=phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=icon_grid.num_levels,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"Koff": dims.KDim},
    )

    assert helpers.dallclose(
        surface_pressure.asnumpy()[:, icon_grid.num_levels],
        surface_pressure_ref,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.JABW_EXPERIMENT,
    ],
)
def test_diagnose_pressure(experiment, data_provider, icon_grid, backend, metrics_savepoint):
    ddqz_z_full = metrics_savepoint.ddqz_z_full()

    diagnostics_reference_savepoint = data_provider.from_savepoint_diagnostics_initial()
    virtual_temperature = diagnostics_reference_savepoint.temperature()
    surface_pressure = diagnostics_reference_savepoint.pressure_sfc()

    pressure_ifc_ref = diagnostics_reference_savepoint.pressure_ifc().asnumpy()
    pressure_ref = diagnostics_reference_savepoint.pressure().asnumpy()

    pressure = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, backend=backend
    )
    cell_domain = h_grid.domain(dims.CellDim)

    pressure_ifc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=float, extend={dims.KDim: 1}, backend=backend
    )

    pressure_ifc.ndarray[:, -1] = surface_pressure.ndarray

    diagnose_pressure.diagnose_pressure.with_backend(backend)(
        ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        pressure,
        pressure_ifc,
        phy_const.GRAV_O_RD,
        horizontal_start=0,
        horizontal_end=icon_grid.end_index(cell_domain(h_grid.Zone.END)),
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert helpers.dallclose(pressure_ifc_ref, pressure_ifc.asnumpy())

    assert helpers.dallclose(
        pressure_ref,
        pressure.asnumpy(),
    )
