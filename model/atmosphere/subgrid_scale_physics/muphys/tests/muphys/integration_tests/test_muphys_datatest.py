# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    component as muphys_component,
    config as muphys_config,
    data as muphys_data,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES, Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import run_full_muphys
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.states.data import QC, QG, QI, QR, QS, QV
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_types
    from icon4py.model.testing import serialbox as sb


# Validation of the muphys granule against Fortran ICON: the aes-graupel-init/exit
# savepoints are written around the mig block in aes_phy_main (cloud_mig =
# satad + graupel + satad), which is exactly the composition MuphysComponent runs
# in both setup_muphys execution modes. Inputs are captured after ICON's
# dyn2phy negative-tracer clipping, so the granule sees identical state.
#
# ICON only computes the scheme on levels jks_cloudy..nlev (zmaxcloudy cutoff)
# while the granule runs the full column, hence the comparisons are restricted
# to the cloudy levels and the levels above are separately checked to produce
# (near-)zero tendencies.
#
# The granule runs with the AES graupel scheme, the port of the ICON
# formulation (mo_aes_graupel.f90) that generates the reference data, so
# near-roundoff agreement is expected. Remaining known deviations:
#   - Fortran clamps tendencies to full depletion (MAX(-q/dt), mo_cloud_mig.f90)
#     while the granule reports raw (new-old)/dt; differs only where the scheme
#     drives a species below zero.
#   - Fortran cvd is computed as cpd - rd (1 ULP off the icon4py literal 717.60);
#     likewise tfrz_hom/tfrz_het2 are computed as tmelt-37/tmelt-25 (1 ULP below
#     the icon4py literals 236.15/248.15) — both only matter at exact thresholds.
#   - Fortran ice_sticking carries a cia tuning factor (cloud_mig_nml, default
#     1.0 and not set in the experiment, hence a no-op here).
@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.single_precision_ready
@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.EXCLAIM_APE_AES],
    ids=lambda experiment: experiment.name,
)
@pytest.mark.parametrize(
    ("date", "single_program"),
    [
        *(
            pytest.param(d, False, id=f"{d[:19]}-separate")
            for d in definitions.Experiments.EXCLAIM_APE_AES.dates
        ),
        pytest.param(
            definitions.Experiments.EXCLAIM_APE_AES.dates[0],
            True,
            id=f"{definitions.Experiments.EXCLAIM_APE_AES.dates[0][:19]}-single",
        ),
    ],
)
def test_muphys_granule(
    date: str,
    single_program: bool,
    *,
    experiment: definitions.Experiment,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_types.IconGrid,
    backend: gtx_typing.Backend,
) -> None:
    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    exit_savepoint = data_provider.from_savepoint_muphys_exit(date=date)

    dtime = experiment.config.driver.dtime.total_seconds()
    # numpy index of the first level ICON computes the scheme on (Fortran jks_cloudy is 1-based)
    jks = init_savepoint.jks_cloudy() - 1
    # Measured max deviations against ICON (gtfn_cpu): tracers 1.9e-16, temperature
    # 2.8e-13. The tolerances leave ~500x / ~350x headroom for other backends.
    tracer_atol = 1e-13 if test_utils.wp_is_dp else 8e-7
    temperature_atol = 1e-10 if test_utils.wp_is_dp else 0.004

    # tendencies are (state difference)/dt, so their tolerances scale with 1/dt
    tracer_tend_atol = tracer_atol / dtime
    temperature_tend_atol = temperature_atol / dtime if test_utils.wp_is_dp else 1e-7

    muphys_configuration = muphys_config.MuphysConfig()
    state = {
        "dz": init_savepoint.dz(),
        "te": init_savepoint.temperature(),
        "p": init_savepoint.pressure(),
        "rho": init_savepoint.rho(),
        "qv": init_savepoint.qv(),
        "qc": init_savepoint.qc(),
        "qr": init_savepoint.qr(),
        "qs": init_savepoint.qs(),
        "qi": init_savepoint.qi(),
        "qg": init_savepoint.qg(),
    }
    initial_state = {name: field.asnumpy().copy() for name, field in state.items()}
    muphys_program = run_full_muphys.setup_muphys(
        ncells=icon_grid.num_cells,
        nlev=icon_grid.num_levels,
        dt=dtime,
        qnc=muphys_configuration.qnc,
        backend=backend,
        single_program=single_program,
    )
    component = muphys_component.MuphysComponent(
        grid=icon_grid,
        dtime=datetime.timedelta(seconds=dtime),
        qnc=muphys_configuration.qnc,
        backend=backend,
        # The default Component setup is the separate-program mode.
        step=muphys_program if single_program else None,
    )
    outputs = component(state, datetime.datetime.fromisoformat(date))
    fields_out = cast(
        "dict[str, fa.CellKField[ta.wpfloat]]", outputs
    )  # component returns gt4py fields at runtime; public signature widens them to DataField
    assert outputs.keys() == component.outputs_properties.keys()
    for name, field in state.items():
        np.testing.assert_array_equal(field.asnumpy(), initial_state[name], err_msg=name)

    # Keep the required in-place aliasing, but use independent buffers so the
    # direct run cannot change the Component's inputs or outputs.
    allocator = model_backends.get_allocator(backend)
    cell_k_dims = (dims.CellDim, dims.KDim)
    direct_t = gtx.as_field(cell_k_dims, initial_state["te"], allocator=allocator)
    direct_q = Q(
        **{
            s: gtx.as_field(cell_k_dims, initial_state[f"q{s}"], allocator=allocator)
            for s in SPECIES
        }
    )
    direct_precip = {
        port: gtx.zeros(state["te"].domain, dtype=ta.wpfloat, allocator=allocator)
        for port in muphys_data.PRECIP_PORTS
    }
    muphys_program(
        dz=state["dz"],
        te=direct_t,
        p=state["p"],
        rho=state["rho"],
        q_in=direct_q,
        t_out=direct_t,
        q_out=direct_q,
        **direct_precip,
    )

    cell_domain = h_grid.domain(dims.CellDim)
    cell_start = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    cells = slice(cell_start, cell_end)
    # Check all seven tendency conversions, including zero boundary/halo rows.
    tendencies: list[tuple[str, fa.CellKField[ta.wpfloat]]] = [
        ("temperature", direct_t),
        *((f"q{species}", getattr(direct_q, species)) for species in SPECIES),
    ]
    for name, updated in tendencies:
        old = initial_state["te" if name == "temperature" else name]
        expected = np.zeros_like(old)
        expected[cells, :] = (updated.asnumpy()[cells, :] - old[cells, :]) / dtime
        np.testing.assert_array_equal(fields_out[f"tend_{name}"].asnumpy(), expected, err_msg=name)

    # ICON saves only aggregate surface diagnostics. Verify the full pflx
    # profile and each surface diagnostic against the direct muphys call.
    np.testing.assert_array_equal(fields_out["pflx"].asnumpy(), direct_precip["pflx"].asnumpy())
    for name in ("pr", "ps", "pi", "pg", "pre"):
        np.testing.assert_array_equal(
            fields_out[name].asnumpy()[:, -1],
            direct_precip[name].asnumpy()[:, -1],
        )

    for name, tracer_index in (
        ("tend_qv", QV),
        ("tend_qc", QC),
        ("tend_qr", QR),
        ("tend_qs", QS),
        ("tend_qi", QI),
        ("tend_qg", QG),
    ):
        test_utils.assert_dallclose(
            getattr(direct_q, name.removeprefix("tend_q")).asnumpy()[cells, jks:],
            exit_savepoint.tracer(tracer_index).asnumpy()[cells, jks:],
            atol=tracer_atol,
            err_msg=f"{name.removeprefix('tend_')} in cloud",
        )
        actual = fields_out[name].asnumpy()
        # above the cloudy region ICON does not run the scheme; the full-column
        # granule must produce (near-)zero tendencies there
        test_utils.assert_dallclose(
            actual[cells, :jks], 0.0, atol=tracer_tend_atol, err_msg=f"{name} above cloud"
        )

    test_utils.assert_dallclose(
        direct_t.asnumpy()[cells, jks:],
        exit_savepoint.temperature().asnumpy()[cells, jks:],
        atol=temperature_atol,
        err_msg="temperature in cloud",
    )
    tend_ta_actual = fields_out["tend_temperature"].asnumpy()
    test_utils.assert_dallclose(
        tend_ta_actual[cells, :jks],
        0.0,
        atol=temperature_tend_atol,
        err_msg="tend_temperature above cloud",
    )

    # surface precip: the granule keeps the surface value in the last level; ICON
    # only stores the aggregated prm_field diagnostics (rsfl = rain,
    # ssfl = ice + snow + graupel, pr = total, ufcs = energy flux)
    rain = fields_out["pr"].asnumpy()[:, -1]
    ice = fields_out["pi"].asnumpy()[:, -1]
    snow = fields_out["ps"].asnumpy()[:, -1]
    graupel = fields_out["pg"].asnumpy()[:, -1]
    energy_flux = fields_out["pre"].asnumpy()[:, -1]

    test_utils.assert_dallclose(
        rain,
        exit_savepoint.rsfl().asnumpy(),
        atol=1e-10 if test_utils.wp_is_dp else 9e-8,
        err_msg="rsfl (rain)",
    )
    test_utils.assert_dallclose(
        ice + snow + graupel,
        exit_savepoint.ssfl().asnumpy(),
        atol=1e-10,
        err_msg="ssfl (ice + snow + graupel)",
    )
    test_utils.assert_dallclose(
        rain + ice + snow + graupel,
        exit_savepoint.pr().asnumpy(),
        atol=1e-10 if test_utils.wp_is_dp else 9e-8,
        err_msg="pr (total precipitation)",
    )
    # ufcs is a large-magnitude flux whose error scales with it: check relative error only
    test_utils.assert_dallclose(
        energy_flux,
        exit_savepoint.ufcs().asnumpy(),
        rtol=1e-12 if test_utils.wp_is_dp else 5e-4,
        err_msg="ufcs (energy flux)",
    )
