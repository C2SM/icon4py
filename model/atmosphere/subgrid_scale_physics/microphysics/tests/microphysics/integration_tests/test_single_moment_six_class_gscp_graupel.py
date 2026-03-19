# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import math

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    microphysics_options as mphys_options,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.atmosphere.advection import advection, advection_states
from icon4py.model.common import dimension as dims, type_alias as ta, model_backends
from icon4py.model.common.grid import vertical as v_grid, simple as simple_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
# from matplotlib import colormaps as cmaps
import colormaps as cmaps
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.testing import serialbox as sb


def setup_plot_style():
    """论文级绘图风格"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": True,
    })


def setup_contour_style():
    """论文级绘图风格"""
    plt.rcParams.update({
        "font.size": 12.5,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 13,
        "axes.titlesize": 16.4,
        "axes.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": True,
    })


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, model_top_height",
    [
        (definitions.Experiments.WEISMAN_KLEMP_TORUS, 30000.0),
    ],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
def test_graupel(
    experiment: definitions.Experiments,
    model_top_height: ta.wpfloat,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: icon_grid.IconGrid,
    lowest_layer_thickness: ta.wpfloat,
    backend: gtx_typing.Backend,
):
    pytest.xfail("Tolerances have increased with new ser_data, need to check with @ongchia")
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    dtime = entry_savepoint.dtime()

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=entry_savepoint.qr(),
        qi=entry_savepoint.qi(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=entry_savepoint.rho(), vn=None, w=None, exner=None, theta_v=None
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=entry_savepoint.temperature(),
        virtual_temperature=None,
        pressure=entry_savepoint.pressure(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,
        ice_stickeff_min=0.01,
        power_law_coeff_for_ice_mean_fall_speed=1.25,
        exponent_for_density_factor_in_ice_sedimentation=0.3,
        power_law_coeff_for_snow_fall_speed=20.0,
        rain_mu=0.0,
        rain_n0=1.0,
        snow2graupel_riming_coeff=0.5,
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qnc = entry_savepoint.qnc()

    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qv_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qc_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qr_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qi_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qs_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qg_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )

    graupel_microphysics.run(
        dtime,
        prognostic_state.rho,
        diagnostic_state.temperature,
        diagnostic_state.pressure,
        tracer_state.qv,
        tracer_state.qc,
        tracer_state.qr,
        tracer_state.qi,
        tracer_state.qs,
        tracer_state.qg,
        qnc,
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qr_tendency,
        qi_tendency,
        qs_tendency,
        qg_tendency,
    )

    new_temperature = (
        entry_savepoint.temperature().asnumpy() + temperature_tendency.asnumpy() * dtime
    )
    new_qv = entry_savepoint.qv().asnumpy() + qv_tendency.asnumpy() * dtime
    new_qc = entry_savepoint.qc().asnumpy() + qc_tendency.asnumpy() * dtime
    new_qr = entry_savepoint.qr().asnumpy() + qr_tendency.asnumpy() * dtime
    new_qi = entry_savepoint.qi().asnumpy() + qi_tendency.asnumpy() * dtime
    new_qs = entry_savepoint.qs().asnumpy() + qs_tendency.asnumpy() * dtime
    new_qg = entry_savepoint.qg().asnumpy() + qg_tendency.asnumpy() * dtime

    assert test_utils.dallclose(
        new_temperature,
        exit_savepoint.temperature().asnumpy(),
    )
    assert test_utils.dallclose(
        new_qv,
        exit_savepoint.qv().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qc,
        exit_savepoint.qc().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qr,
        exit_savepoint.qr().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qi,
        exit_savepoint.qi().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qs,
        exit_savepoint.qs().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qg,
        exit_savepoint.qg().asnumpy(),
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        graupel_microphysics.rain_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.rain_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.snow_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.snow_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.graupel_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.graupel_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.ice_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.ice_flux().asnumpy()[:],
        atol=9.0e-11,
    )


qnc = data_alloc.zero_field(
    grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator
)
qnc_ndarray = qnc.ndarray
TIMESTEPS = 3600
DTIME = 1.0
def generate_initial_condition():
    ...
def generate_vertical_grid(num_levels, backend):
    ...
    
@pytest.mark.parametrize(
    "num_levels, rain_n0, n_c", [(50, 100.0, 30.e6)]
)
def test_collision_coalescence(
    grid,
    num_levels,
    rain_n0,
    n_c,
    backend,
):
    vertical_params = generate_vertical_grid(num_levels, backend)
    
    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,     # type: ignore
        rain_n0=rain_n0,
    )
    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,                                                                     # type: ignore
        vertical_params=vertical_params,
        backend=backend,
    )

    tracer_state = generate_initial_condition(vertical_params, grid, backend)                                 # type: ignore
    qnc_ndarray[:] = n_c                                                                       # type: ignore
    for _ in range(TIMESTEPS):
        graupel_microphysics.run(DTIME, tracer_state, qnc)                                                                     # type: ignore

def update_flux():
    ...

def test_kinematic_driver(
    grid,
    backend,
):
    advection_config = advection.AdvectionConfig(
        horizontal_advection_type=advection.HorizontalAdvectionType.NO_ADVECTION,
        horizontal_advection_limiter=advection.HorizontalAdvectionLimiter.NO_LIMITER,
        vertical_advection_type=advection.VerticalAdvectionType.UPWIND_1ST_ORDER,
        vertical_advection_limiter=advection.VerticalAdvectionLimiter.NO_LIMITER,
    )
    advection_granule = advection.convert_config_to_advection(
        config=advection_config,
        grid=grid,
        backend=backend,
    )                                                                                               # type: ignore
    prep_adv_state, tracer_state = generate_initial_condition(vertical_params, grid, backend)                                 # type: ignore
    for _ in range(TIMESTEPS):
        graupel_microphysics.run(DTIME, tracer_state, qnc)                                                                     # type: ignore
        advection_granule.run(DTIME, tracer_state, prep_adv_state)
        update_flux(prep_adv_state)                                                                                           # type: ignore

@pytest.mark.parametrize(
    "num_levels", [50]
)
def test_collision_coalescence2(
    num_levels: int,
    backend: gtx_typing.Backend,
):
    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)
    initial_liquid_content = ta.wpfloat("1.0") # g m-3
    top_height = ta.wpfloat("5000.0")
    dz = top_height / num_levels
    dtime = ta.wpfloat("1.0")
    
    grid = simple_grid.simple_grid(allocator=allocator, num_levels=num_levels)
    
    vertical_config = v_grid.VerticalGridConfig(
        num_levels,
        lowest_layer_thickness=dz,
        model_top_height=top_height,
    )

    vct_a = xp.linspace(top_height, 0.0, num_levels+1, dtype=ta.wpfloat)
    z_mc = 0.5 * (vct_a[:-1] + vct_a[1:])
    vct_b = xp.zeros(num_levels+1, dtype=ta.wpfloat)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), vct_a, allocator=allocator),
        vct_b=gtx.as_field((dims.KDim,), vct_b, allocator=allocator),
    )

    ddqz_z_full = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator)
    ddqz_z_full_ndarray = ddqz_z_full.ndarray
    ddqz_z_full_ndarray[:,:] = dz
    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=ddqz_z_full,
    )

    tracer_state = tracers.TracerState(
        qv=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qc=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qr=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qi=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qs=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qg=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        virtual_temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        pressure=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    qnc = data_alloc.zero_field(
        grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator
    )

    temperature_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qv_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qc_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qr_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qi_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qs_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qg_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )

    rho_ndarray = prognostic_state.rho.ndarray
    temperature_ndarray = diagnostic_state.temperature.ndarray
    pressure_ndarray = diagnostic_state.pressure.ndarray
    qv_ndarray = tracer_state.qv.ndarray
    qc_ndarray = tracer_state.qc.ndarray
    qi_ndarray = tracer_state.qi.ndarray
    qr_ndarray = tracer_state.qr.ndarray
    qs_ndarray = tracer_state.qs.ndarray
    qg_ndarray = tracer_state.qg.ndarray
    qnc_ndarray = qnc.ndarray
    temperature_tendency_ndarray = temperature_tendency.ndarray
    qv_tendency_ndarray = qv_tendency.ndarray
    qc_tendency_ndarray = qc_tendency.ndarray
    qr_tendency_ndarray = qr_tendency.ndarray
    qi_tendency_ndarray = qi_tendency.ndarray
    qs_tendency_ndarray = qs_tendency.ndarray
    qg_tendency_ndarray = qg_tendency.ndarray

    qnc_ndarray[:] = 30.e6 #138.e6 #50.0e6

    rho_ndarray[:,:] = ta.wpfloat("1.2") # kg m-3
    temperature_ndarray[:,:] = ta.wpfloat("293.15") # K
    pressure_ndarray[:,:] = ta.wpfloat("101325.0") # Pa
    qv_ndarray[:,:] = ta.wpfloat("0.01") # kg kg-1
    cloud_boundary_index = xp.argmin(xp.abs(z_mc - 1000.0)) # height of cloud boundary at 2000 m
    if z_mc[cloud_boundary_index] < 1000.0:
        cloud_boundary_index -= 1

    # print()
    # for k in range(num_levels):
    #     print(k, z_mc[k], temperature_ndarray[0, k], qc_ndarray[0, k])
    print()
    for k in range(num_levels):
        print("initial profile: ", k, z_mc[k], xp.mean(ddqz_z_full_ndarray[:,k]), xp.mean(temperature_ndarray[:, k]), xp.mean(rho_ndarray[:, k]), xp.mean(qv_ndarray[:, k]), xp.mean(qc_ndarray[:, k]))

    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,
        rain_mu=ta.wpfloat("0.0"),
        rain_n0=ta.wpfloat("100.0"),
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    print()

    reference_time = xp.array([25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60])
    reference_rate = xp.array([0.2, 1.5, 6.1, 11.8, 13.0, 11.0, 8.3, 6.05, 4.5, 3.25, 2.6, 2.0, 1.7, 1.25, 1.12])
    
    prep_time_default, prep_rate_default = [], []
    qc_ndarray[:, :cloud_boundary_index+1] = initial_liquid_content / 1.2 / 1000.0 # convert from g m-3 to kg kg-1
    qr_ndarray[:, :] = ta.wpfloat("0.0")
    for i in range(60*60):
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
            do_warm_cloud=True,
        )
        temperature_ndarray += temperature_tendency_ndarray * dtime
        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        qr_ndarray += qr_tendency_ndarray * dtime
        qi_ndarray += qi_tendency_ndarray * dtime
        qs_ndarray += qs_tendency_ndarray * dtime
        qg_ndarray += qg_tendency_ndarray * dtime

        # print("time step", i)
        # print("max t, qv, qc tendencies:", xp.max(xp.abs(temperature_tendency.ndarray)), xp.max(xp.abs(qv_tendency.ndarray)), xp.max(xp.abs(qc_tendency.ndarray)))
        # print("max qr, qi, qs, qg tendencies:", xp.max(xp.abs(qr_tendency.ndarray)), xp.max(xp.abs(qi_tendency.ndarray)), xp.max(xp.abs(qs_tendency.ndarray)), xp.max(xp.abs(qg_tendency.ndarray)))
        # print("precipitation flux: ", xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0)) # / 1000 * 3600 * 1000, convert from kg m-2 s-1 to mm h-1, kg -> m, s-1 to h-1, mm -> m
        # print("===end===")
        
        
        prep_time_default.append((i + 1) * dtime / 60.0)
        prep_rate_default.append(xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0))

        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qr_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qi_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qs_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qg_tendency_ndarray[:,:] = ta.wpfloat("0.0")
    
    qnc_ndarray[:] = 10.e6 #138.e6 #50.0e6
    prep_time_10qnc, prep_rate_10qnc = [], []
    qc_ndarray[:, :cloud_boundary_index+1] = initial_liquid_content / 1.2 / 1000.0 # convert from g m-3 to kg kg-1
    qr_ndarray[:, :] = ta.wpfloat("0.0")
    for i in range(60*60):
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
            do_warm_cloud=True,
        )
        temperature_ndarray += temperature_tendency_ndarray * dtime
        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        qr_ndarray += qr_tendency_ndarray * dtime
        qi_ndarray += qi_tendency_ndarray * dtime
        qs_ndarray += qs_tendency_ndarray * dtime
        qg_ndarray += qg_tendency_ndarray * dtime

        prep_time_10qnc.append((i + 1) * dtime / 60.0)
        prep_rate_10qnc.append(xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0))

        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qr_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qi_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qs_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qg_tendency_ndarray[:,:] = ta.wpfloat("0.0")
    
    qnc_ndarray[:] = 50.e6 #138.e6 #50.0e6
    prep_time_50qnc, prep_rate_50qnc = [], []
    qc_ndarray[:, :cloud_boundary_index+1] = initial_liquid_content / 1.2 / 1000.0 # convert from g m-3 to kg kg-1
    qr_ndarray[:, :] = ta.wpfloat("0.0")
    for i in range(60*60):
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
            do_warm_cloud=True,
        )
        temperature_ndarray += temperature_tendency_ndarray * dtime
        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        qr_ndarray += qr_tendency_ndarray * dtime
        qi_ndarray += qi_tendency_ndarray * dtime
        qs_ndarray += qs_tendency_ndarray * dtime
        qg_ndarray += qg_tendency_ndarray * dtime

        prep_time_50qnc.append((i + 1) * dtime / 60.0)
        prep_rate_50qnc.append(xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0))

        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qr_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qi_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qs_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qg_tendency_ndarray[:,:] = ta.wpfloat("0.0")
    
    qnc_ndarray[:] = 30.e6 #138.e6 #50.0e6
    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,
        rain_mu=ta.wpfloat("0.0"),
        rain_n0=ta.wpfloat("1.0"),
    )
    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )
    prep_time_1n0, prep_rate_1n0 = [], []
    qc_ndarray[:, :cloud_boundary_index+1] = initial_liquid_content / 1.2 / 1000.0 # convert from g m-3 to kg kg-1
    qr_ndarray[:, :] = ta.wpfloat("0.0")
    for i in range(60*60):
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
            do_warm_cloud=True,
        )
        temperature_ndarray += temperature_tendency_ndarray * dtime
        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        qr_ndarray += qr_tendency_ndarray * dtime
        qi_ndarray += qi_tendency_ndarray * dtime
        qs_ndarray += qs_tendency_ndarray * dtime
        qg_ndarray += qg_tendency_ndarray * dtime

        prep_time_1n0.append((i + 1) * dtime / 60.0)
        prep_rate_1n0.append(xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0))

        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qr_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qi_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qs_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qg_tendency_ndarray[:,:] = ta.wpfloat("0.0")
    
    setup_plot_style()
    
    def lineplot(time, input_data, input_label, input_style, input_color):
        plt.close()
        fig, ax = plt.subplots(figsize=(6, 4.2))
        
        for i in range(len(input_data)):
            sns.lineplot(
                data=pd.DataFrame(data={"Time": time, "Prep. rate": input_data[i]}),
                x="Time",
                y="Prep. rate",
                color=input_color[i],
                label=input_label[i],
                linewidth=1.8,
                linestyle=input_style[i],
                ax=ax,
                zorder=1,
            )
        sns.scatterplot(
            data=pd.DataFrame(data={"Time": reference_time, "Prep. rate": reference_rate}),
            x="Time",
            y="Prep. rate",
            color="#000000",#"#e2764e",
            label="Reference$^{1}$",
            s=30,
            edgecolor="white",
            linewidth=1.2,
            ax=ax,
            zorder=2,
        )
        ax.spines['bottom'].set_alpha(0.8)
        ax.spines['left'].set_alpha(0.8)
        ax.spines['top'].set_alpha(0.8)
        ax.spines['right'].set_alpha(0.8)
        ax.set_ylim(-0.1, 20)
        # ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-0.7, 60.5)
        ax.set_xticks(xp.linspace(0, 60, 13))
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Prep. rate (mm h$^{-1}$)")
        ax.grid(True, linestyle="--", alpha=0.0)
        # ax.set_title("Surface pressure error evolution")
        plt.tight_layout()
        plt.legend(loc="upper left", fontsize=12)
        plt.savefig("rain_shaft_test.png", dpi=300)

    LINE_COLORS = ["#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    lineplot(
        prep_time_default,
        (
            prep_rate_default,
            prep_rate_10qnc,
            prep_rate_50qnc,
            prep_rate_1n0,
        ),
        (
            "Default",
            "N$_c$ = 10 cm$^{-3}$",
            "N$_c$ = 50 cm$^{-3}$",
            "rain_n0 = 1",
        ),
        (
            "solid",
            "dashed",
            "dashed",
            "dotted",
        ),
        LINE_COLORS,
    )



@pytest.mark.parametrize(
    "num_levels", [163]
)
def test_shipway_hall(
    num_levels: int,
    backend: gtx_typing.Backend,
):
    allocator = model_backends.get_allocator(backend)
    xp = data_alloc.import_array_ns(allocator)
    top_height = ta.wpfloat("3260.0")
    dz = top_height / num_levels
    dtime = ta.wpfloat("1.0")
    
    grid = simple_grid.simple_grid(allocator=allocator, num_levels=num_levels)
    
    vertical_config = v_grid.VerticalGridConfig(
        num_levels,
        lowest_layer_thickness=dz,
        model_top_height=top_height,
    )

    vct_a = xp.linspace(top_height, 0.0, num_levels+1, dtype=ta.wpfloat)
    z_mc = 0.5 * (vct_a[:-1] + vct_a[1:])
    vct_b = xp.zeros(num_levels+1, dtype=ta.wpfloat)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), vct_a, allocator=allocator),
        vct_b=gtx.as_field((dims.KDim,), vct_b, allocator=allocator),
    )

    ddqz_z_full = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator)
    ddqz_z_full_ndarray = ddqz_z_full.ndarray
    ddqz_z_full_ndarray[:,:] = dz
    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=ddqz_z_full,
    )

    tracer_state = tracers.TracerState(
        qv=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qc=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qr=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qi=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qs=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        qg=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        vn=None,
        w=None,
        exner=None,
        theta_v=None,
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        virtual_temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        pressure=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    advection_diagnostic_state = advection_states.AdvectionDiagnosticState(
        airmass_now=data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim, allocator=allocator),
        airmass_new=data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim, allocator=allocator),
        grf_tend_tracer=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
        hfl_tracer=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        vfl_tracer=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator, extend={dims.KDim: 1}),
    )
    prep_adv = advection_states.AdvectionPrepAdvState(
        vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_ic=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator, extend={dims.KDim: 1}),
    )
    
    qnc = data_alloc.zero_field(
        grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator
    )

    temperature_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qv_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qc_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qr_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qi_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qs_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )
    qg_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
    )

    p_tracer_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)

    rho_ndarray = prognostic_state.rho.ndarray
    temperature_ndarray = diagnostic_state.temperature.ndarray
    potential_temperature_ndarray = xp.zeros((temperature_ndarray.shape[0], num_levels), dtype=ta.wpfloat)
    assert potential_temperature_ndarray.shape == temperature_ndarray.shape
    pressure_ndarray = diagnostic_state.pressure.ndarray
    qv_ndarray = tracer_state.qv.ndarray
    qc_ndarray = tracer_state.qc.ndarray
    qi_ndarray = tracer_state.qi.ndarray
    qr_ndarray = tracer_state.qr.ndarray
    qs_ndarray = tracer_state.qs.ndarray
    qg_ndarray = tracer_state.qg.ndarray
    qnc_ndarray = qnc.ndarray
    temperature_tendency_ndarray = temperature_tendency.ndarray
    qv_tendency_ndarray = qv_tendency.ndarray
    qc_tendency_ndarray = qc_tendency.ndarray
    qr_tendency_ndarray = qr_tendency.ndarray
    qi_tendency_ndarray = qi_tendency.ndarray
    qs_tendency_ndarray = qs_tendency.ndarray
    qg_tendency_ndarray = qg_tendency.ndarray
    mass_flx_ic_ndarray = prep_adv.mass_flx_ic.ndarray
    p_tracer_new_ndarray = p_tracer_new.ndarray

    Rd = ta.wpfloat("287.04")
    Cp = ta.wpfloat("1004.64")
    P0 = ta.wpfloat("100000.0")
    qnc_ndarray[:] = 50.0e6
    pressure_ndarray[:,:] = ta.wpfloat("101325.0") # Pa
    pressure_ndarray[:,-1] = ta.wpfloat("80000.0") # Pa
    for k in range(num_levels):
        pressure_ndarray[:,k] = ta.wpfloat("101325.0") * xp.exp(-z_mc[k] / 13810.0)
        if z_mc[k] <= 740.0:
            potential_temperature_ndarray[:,k] = ta.wpfloat("297.9")
            # temperature_ndarray[:,k] = ta.wpfloat("297.9")
            qv_ndarray[:,k] = ( 0.015 * (740.0 - z_mc[k]) + 0.0138 * z_mc[k] ) / 740.0
        else:
            potential_temperature_ndarray[:,k] = ( 297.9 * (3260.0 - z_mc[k]) + 312.66 * (z_mc[k] - 740.0) ) / (3260.0 - 740.0)
            # temperature_ndarray[:,k] = ( 297.9 * (3260.0 - z_mc[k]) + 312.66 * (z_mc[k] - 740.0) ) / (3260.0 - 740.0)
            qv_ndarray[:,k] = ( 0.0138 * (3260.0 - z_mc[k]) + 0.0024 * (z_mc[k] - 740.0) ) / (3260.0 - 740.0)
    # qv_ndarray[:,:] = qv_ndarray[:,:] * 2.0
    VERTICAL_WIND_SPEED = ta.wpfloat("3.0")
    mass_flx_ic_ndarray[:,:-1] = VERTICAL_WIND_SPEED
    temperature_ndarray[:,:] = potential_temperature_ndarray[:,:] * (pressure_ndarray[:,:] / P0) ** (Rd / Cp) - 3.0 # adjust temperature profile to make the column unstable
    rho_ndarray[:,:] = pressure_ndarray[:,:] / (Rd * temperature_ndarray[:,:]) #  * (1.0 + qv_ndarray[:,:]) adjust density for water vapor

    VERTICAL_WIND_PERIOD = ta.wpfloat("900.0") # s

    # print()
    # for k in range(num_levels):
    #     print(k, z_mc[k], temperature_ndarray[0, k], qc_ndarray[0, k])
    print()
    for k in range(num_levels):
        print("initial profile: ", k, z_mc[k], xp.mean(ddqz_z_full_ndarray[:,k]), xp.mean(temperature_ndarray[:, k]), xp.mean(rho_ndarray[:, k]), xp.mean(qv_ndarray[:, k]), xp.mean(qc_ndarray[:, k]))

    config = satad.SaturationAdjustmentConfig(
        tolerance=1e-3,
        max_iter=10,
    )

    satad_metric_state = satad.MetricStateSaturationAdjustment(
        ddqz_z_full=ddqz_z_full,
    )
    saturation_adjustment = satad.SaturationAdjustment(
        config=config,
        grid=grid,
        metric_state=satad_metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )
    
    graupel_config = graupel.SingleMomentSixClassIconGraupelConfig(
        liquid_autoconversion_option=mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG,
        rain_mu=ta.wpfloat("0.0"),
        rain_n0=ta.wpfloat("1.0"),
    )

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    advection_config = advection.AdvectionConfig(
        horizontal_advection_type=advection.HorizontalAdvectionType.NO_ADVECTION,
        horizontal_advection_limiter=advection.HorizontalAdvectionLimiter.NO_LIMITER,
        vertical_advection_type=advection.VerticalAdvectionType.UPWIND_1ST_ORDER,
        vertical_advection_limiter=advection.VerticalAdvectionLimiter.NO_LIMITER,
    )
    advection_granule = advection.convert_config_to_advection(
        config=advection_config,
        grid=grid,
        interpolation_state=advection_states.AdvectionInterpolationState(
            geofac_div=None,
            rbf_vec_coeff_e=None,
            pos_on_tplane_e_1=None,
            pos_on_tplane_e_2=None,
        ),
        least_squares_state=advection_states.AdvectionLeastSquaresState(
            lsq_pseudoinv_1=None,
            lsq_pseudoinv_2=None,
        ),
        metric_state=advection_states.AdvectionMetricState(
            deepatmo_divh=data_alloc.constant_field(grid, 1.0 / dz, dims.KDim, allocator=allocator),
            deepatmo_divzl=data_alloc.constant_field(grid, 1.0 / dz, dims.KDim, allocator=allocator),
            deepatmo_divzu=data_alloc.constant_field(grid, 1.0 / dz, dims.KDim, allocator=allocator),
            ddqz_z_full=ddqz_z_full,
        ),
        edge_params=None,
        cell_params=None,
        even_timestep=False,
        backend=backend,
    )
    
    print()
    timesteps = 60 * 30 + 1
    data_qv = xp.zeros((num_levels,timesteps), dtype=ta.wpfloat)
    data_qc = xp.zeros((num_levels,timesteps), dtype=ta.wpfloat)
    data_qr = xp.zeros((num_levels,timesteps), dtype=ta.wpfloat)
    data_time = xp.zeros(timesteps, dtype=ta.wpfloat)
    data_z = xp.array(z_mc, copy=True)
    for i in range(timesteps):
        data_qv[:,i] = xp.mean(qv_ndarray, axis=0)
        data_qc[:,i] = xp.mean(qc_ndarray, axis=0)
        data_qr[:,i] = xp.mean(qr_ndarray, axis=0)
        data_time[i] = i * dtime / 60.0
        graupel_microphysics.run(
            dtime,
            prognostic_state.rho,
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            tracer_state.qv,
            tracer_state.qc,
            tracer_state.qr,
            tracer_state.qi,
            tracer_state.qs,
            tracer_state.qg,
            qnc,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qr_tendency,
            qi_tendency,
            qs_tendency,
            qg_tendency,
            do_warm_cloud=False,
        )
        # temperature_ndarray += temperature_tendency_ndarray * dtime
        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        qr_ndarray += qr_tendency_ndarray * dtime
        qi_ndarray += qi_tendency_ndarray * dtime
        qs_ndarray += qs_tendency_ndarray * dtime
        qg_ndarray += qg_tendency_ndarray * dtime

        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        saturation_adjustment.run(
            dtime=dtime,
            rho=prognostic_state.rho,
            temperature=diagnostic_state.temperature,
            qv=tracer_state.qv,
            qc=tracer_state.qc,
            temperature_tendency=temperature_tendency,
            qv_tendency=qv_tendency,
            qc_tendency=qc_tendency,
        )

        qv_ndarray += qv_tendency_ndarray * dtime
        qc_ndarray += qc_tendency_ndarray * dtime
        
        # advection_granule.run(
        #     diagnostic_state=advection_diagnostic_state,
        #     prep_adv=prep_adv,
        #     p_tracer_now=diagnostic_state.temperature,
        #     p_tracer_new=p_tracer_new,
        #     dtime=dtime,
        # )
        # temperature_ndarray[:,:] = p_tracer_new_ndarray[:,:]
        # p_tracer_new_ndarray[:,:] = ta.wpfloat("0.0")
        
        advection_granule.run(
            diagnostic_state=advection_diagnostic_state,
            prep_adv=prep_adv,
            p_tracer_now=tracer_state.qv,
            p_tracer_new=p_tracer_new,
            dtime=dtime,
        )
        qv_ndarray[:,:] = p_tracer_new_ndarray[:,:]
        p_tracer_new_ndarray[:,:] = ta.wpfloat("0.0")

        advection_granule.run(
            diagnostic_state=advection_diagnostic_state,
            prep_adv=prep_adv,
            p_tracer_now=tracer_state.qc,
            p_tracer_new=p_tracer_new,
            dtime=dtime,
        )
        qc_ndarray[:,:] = p_tracer_new_ndarray[:,:]
        p_tracer_new_ndarray[:,:] = ta.wpfloat("0.0")

        advection_granule.run(
            diagnostic_state=advection_diagnostic_state,
            prep_adv=prep_adv,
            p_tracer_now=tracer_state.qr,
            p_tracer_new=p_tracer_new,
            dtime=dtime,
        )
        qr_ndarray[:,:] = p_tracer_new_ndarray[:,:]
        p_tracer_new_ndarray[:,:] = ta.wpfloat("0.0")

        # force upper boundary condition:
        qv_ndarray[:,0] = qv_ndarray[:,1]
        if (i + 1.0) * dtime < VERTICAL_WIND_PERIOD:
            mass_flx_ic_ndarray[:,:-1] = VERTICAL_WIND_SPEED * math.sin( math.pi * (i + ta.wpfloat("1.0")) * dtime / VERTICAL_WIND_PERIOD )
        else:
            mass_flx_ic_ndarray[:,:] = ta.wpfloat("0.0")

        if i % 60 == 0:
            # print("time step", i)
            # for k in range(num_levels):
            #     print(k, xp.mean(qv_ndarray[:,k]), xp.mean(mass_flx_ic_ndarray[:,k]))
            # print("===end===")  
            print("time step", i)
            print("max t, qv, qc tendencies:", xp.max(xp.abs(temperature_tendency.ndarray)), xp.max(xp.abs(qv_tendency.ndarray)), xp.max(xp.abs(qc_tendency.ndarray)))
            print("max qr, qi, qs, qg tendencies:", xp.max(xp.abs(qr_tendency.ndarray)), xp.max(xp.abs(qi_tendency.ndarray)), xp.max(xp.abs(qs_tendency.ndarray)), xp.max(xp.abs(qg_tendency.ndarray)))
            print("precipitation flux: ", xp.mean(graupel_microphysics.rain_precipitation_flux.ndarray[:, -1]*3600.0)) # / 1000 * 3600 * 1000, convert from kg m-2 s-1 to mm h-1, kg -> m, s-1 to h-1, mm -> m
            print("max t rho pres qv qc: ", xp.max(temperature_ndarray), xp.max(rho_ndarray), xp.max(pressure_ndarray), xp.max(qv_ndarray), xp.max(qc_ndarray))
            print("min t rho pres qv qc: ", xp.min(temperature_ndarray), xp.min(rho_ndarray), xp.min(pressure_ndarray), xp.min(qv_ndarray), xp.min(qc_ndarray))
            print("===end===")
            
        temperature_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qv_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qc_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qr_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qi_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qs_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        qg_tendency_ndarray[:,:] = ta.wpfloat("0.0")
        
    
    data_time2d, data_z2d = xp.meshgrid(data_time, data_z)
    print("DEBUG PLOT SHAPE: ", data_time2d.shape, data_z2d.shape, data_qc.shape)

    setup_contour_style()
    
    contour_cmap = cmaps.batloww.reversed()
    bnd_scale_list = [0, 0.25, 0.5, 0.75, 1.0]

    def contourplot(input_time, input_z, input_data, title, filename):
        plt.close()
        fig, ax = plt.subplots(figsize=(7, 5.2))

        input_data_max = xp.max(input_data)
        input_data_boundaries = xp.linspace(0.0, input_data_max, 100)
        input_data_tboundaries = [i * input_data_max for i in bnd_scale_list] 
        input_data_norm = colors.BoundaryNorm(input_data_boundaries, contour_cmap.N, clip=True)
        cp = ax.contourf(input_time, input_z, input_data, cmap=contour_cmap, levels=input_data_boundaries, norm=input_data_norm)
        cb = plt.colorbar(
            cp,
            ax=ax,
            orientation="horizontal",
            pad=0.18,
            ticks=input_data_tboundaries,
        )
        
        # cb_ticks = cb.ax.get_yticks().tolist()
        # for i, item in enumerate(cb_ticks):
        #     cb_ticks[i] = round(item, 1) # .label.get_text()
        # cb.ax.yaxis.set_ticklabels(cb_ticks)
        # fmt = mticker.ScalarFormatter(useMathText=True)
        # fmt.set_powerlimits((0, 0))
        # fmt.set_scientific(True)
        # fmt.set_useOffset(True)
        # cb.ax.yaxis.set_major_formatter(fmt)
        cb.formatter.set_powerlimits((0, 0))
        
        ax.set_title(title)
        ax.set_xlim(-0.1, 30)
        ax.set_ylim(-1, 2000)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("z (m)")
        plt.savefig(filename, format="png", dpi=300)
        plt.clf()

    contourplot(data_time2d, data_z2d, data_qv*1000.0, "qv (g kg$^{-1})$", "kid_qv.png")
    contourplot(data_time2d, data_z2d, data_qc*1000.0, "qc (g kg$^{-1})$", "kid_qc.png")
    contourplot(data_time2d, data_z2d, data_qr*1000.0, "qr (g kg$^{-1})$", "kid_qr.png")