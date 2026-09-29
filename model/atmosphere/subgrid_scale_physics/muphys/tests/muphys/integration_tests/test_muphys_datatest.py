# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    component as muphys_component,
    config as muphys_config,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, run_full_muphys
from icon4py.model.common import dimension as dims, model_backends
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
@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.EXCLAIM_APE_AES],
    ids=lambda experiment: experiment.name,
)
@pytest.mark.parametrize(
    ("date", "single_program"),
    [
        pytest.param("2008-09-01T00:00:00.000", False, id="2008-09-01T00:00:00-separate"),
        pytest.param("2008-09-01T00:05:00.000", False, id="2008-09-01T00:05:00-separate"),
        pytest.param("2008-09-01T00:10:00.000", False, id="2008-09-01T00:10:00-separate"),
        pytest.param("2008-09-01T00:00:00.000", True, id="2008-09-01T00:00:00-single"),
    ],
)
def test_muphys_granule(
    date: str,
    single_program: bool,
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_types.IconGrid,
    backend: gtx_typing.Backend,
) -> None:
    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    exit_savepoint = data_provider.from_savepoint_muphys_exit(date=date)

    dtime = init_savepoint.dtime()
    # numpy index of the first level ICON computes the scheme on (Fortran jks_cloudy is 1-based)
    jks = init_savepoint.jks_cloudy() - 1

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
    assert outputs.keys() == component.outputs_properties.keys()
    for name, field in state.items():
        np.testing.assert_array_equal(field.asnumpy(), initial_state[name], err_msg=name)

    # Keep the required in-place aliasing, but use independent buffers so the
    # direct run cannot change the Component's inputs or outputs.
    direct = common.GraupelOutput.allocate(
        allocator=model_backends.get_allocator(backend),
        domain=gtx.domain({dims.CellDim: icon_grid.num_cells, dims.KDim: icon_grid.num_levels}),
    )
    direct.t.ndarray[...] = state["te"].ndarray
    for species in SPECIES:
        getattr(direct, f"q{species}").ndarray[...] = state[f"q{species}"].ndarray
    muphys_program(
        dz=state["dz"],
        te=direct.t,
        p=state["p"],
        rho=state["rho"],
        q_in=direct.q,
        t_out=direct.t,
        q_out=direct.q,
        pflx=direct.pflx,
        pr=direct.pr,
        ps=direct.ps,
        pi=direct.pi,
        pg=direct.pg,
        pre=direct.pre,
    )

    cell_domain = h_grid.domain(dims.CellDim)
    cell_start = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    cells = slice(cell_start, cell_end)
    # Check all seven tendency conversions, including zero boundary/halo rows.
    for name, updated in (
        ("temperature", direct.t),
        *((f"q{species}", getattr(direct, f"q{species}")) for species in SPECIES),
    ):
        old = initial_state["te" if name == "temperature" else name]
        expected = np.zeros_like(old)
        expected[cells, :] = (updated.asnumpy()[cells, :] - old[cells, :]) / dtime
        np.testing.assert_array_equal(outputs[f"tend_{name}"].asnumpy(), expected, err_msg=name)

    # ICON saves only aggregate surface diagnostics. Verify the full pflx
    # profile and each surface diagnostic against the direct muphys call.
    np.testing.assert_array_equal(outputs["pflx"].asnumpy(), direct.pflx.asnumpy())
    for name in ("pr", "ps", "pi", "pg", "pre"):
        np.testing.assert_array_equal(
            outputs[name].asnumpy()[:, -1],
            getattr(direct, name).asnumpy()[:, -1],
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
            getattr(direct, name.removeprefix("tend_")).asnumpy()[cells, jks:],
            exit_savepoint.tracer(tracer_index).asnumpy()[cells, jks:],
            atol=1e-13,
            err_msg=f"{name.removeprefix('tend_')} in cloud",
        )
        actual = outputs[name].asnumpy()
        # above the cloudy region ICON does not run the scheme; the full-column
        # granule must produce (near-)zero tendencies there
        test_utils.assert_dallclose(
            actual[cells, :jks], 0.0, atol=1e-12, err_msg=f"{name} above cloud"
        )

    test_utils.assert_dallclose(
        direct.t.asnumpy()[cells, jks:],
        exit_savepoint.temperature().asnumpy()[cells, jks:],
        atol=1e-10,
        err_msg="temperature in cloud",
    )
    tend_ta_actual = outputs["tend_temperature"].asnumpy()
    test_utils.assert_dallclose(
        tend_ta_actual[cells, :jks], 0.0, atol=1e-10, err_msg="tend_temperature above cloud"
    )

    # surface precip: the granule keeps the surface value in the last level; ICON
    # only stores the aggregated prm_field diagnostics (rsfl = rain,
    # ssfl = ice + snow + graupel, pr = total, ufcs = energy flux)
    rain = outputs["pr"].asnumpy()[:, -1]
    ice = outputs["pi"].asnumpy()[:, -1]
    snow = outputs["ps"].asnumpy()[:, -1]
    graupel = outputs["pg"].asnumpy()[:, -1]
    energy_flux = outputs["pre"].asnumpy()[:, -1]

    test_utils.assert_dallclose(
        rain, exit_savepoint.rsfl().asnumpy(), atol=1e-10, err_msg="rsfl (rain)"
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
        atol=1e-10,
        err_msg="pr (total precipitation)",
    )
    test_utils.assert_dallclose(
        energy_flux, exit_savepoint.ufcs().asnumpy(), atol=1e-10, err_msg="ufcs (energy flux)"
    )
