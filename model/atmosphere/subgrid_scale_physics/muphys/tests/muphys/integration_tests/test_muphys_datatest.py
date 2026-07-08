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

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    component as muphys_component,
    config as muphys_config,
)
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
# (setup_muphys with single_program=False). Inputs are captured after ICON's
# dyn2phy negative-tracer clipping, so the granule sees identical state.
#
# ICON only computes the scheme on levels jks_cloudy..nlev (zmaxcloudy cutoff)
# while the granule runs the full column, hence the comparisons are restricted
# to the cloudy levels and the levels above are separately checked to produce
# (near-)zero tendencies.
#
# The exit tendencies are the prm_tend accumulators; subtracting the init
# accumulators isolates the mig contribution even if other AES processes ever
# run before mig in this experiment (today they contribute exactly zero).
#
# KNOWN DIVERGENCES vs the icon-nwp tree that generates the reference data (the
# port follows the older muphys C++ reference, icon-nwp carries the newer MPIM
# rain-microphysics revisions); expect real physics differences, not roundoff,
# wherever these paths are active:
#   - cloud_to_rain: accretion kernel is a polynomial in log(rho*qr) in Fortran
#     (mo_aes_graupel.f90:601), a constant AC_KERNEL=5.25 here; the Fortran
#     version also takes rho.
#   - rain_to_vapor: exp-polynomial evaporation in Fortran (mo_aes_graupel.f90:714)
#     vs the older power-law here.
#   - Fortran clamps tendencies to full depletion (MAX(-q/dt), mo_cloud_mig.f90)
#     while the granule reports raw (new-old)/dt.
@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.EXCLAIM_APE_AES],
)
@pytest.mark.parametrize(
    "date",
    ["2008-09-01T00:05:00.000", "2008-09-01T00:10:00.000", "2008-09-01T00:15:00.000"],
)
def test_muphys_granule(
    date: str,
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

    # MuphysConfig().qnc matches the Fortran cloud_num = 50.0e6 m^-3 (mo_cloud_mig.f90)
    component = muphys_component.MuphysComponent(
        ncells=icon_grid.num_cells,
        nlev=icon_grid.num_levels,
        dtime=datetime.timedelta(seconds=dtime),
        qnc=muphys_config.MuphysConfig().qnc,
        backend=backend,
    )

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
    outputs = component(state, datetime.datetime.fromisoformat(date))

    # provisional tolerances assuming matching formulas; the known divergences listed
    # above will show up as real physics differences wherever accretion/evaporation
    # are active — measure on the archive (ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL=true)
    # and either port the newer Fortran formulas or loosen with a documented rationale
    for name, tracer_index in (
        ("tend_qv", QV),
        ("tend_qc", QC),
        ("tend_qr", QR),
        ("tend_qs", QS),
        ("tend_qi", QI),
        ("tend_qg", QG),
    ):
        reference = (
            exit_savepoint.tend_tracer(tracer_index).asnumpy()
            - init_savepoint.tend_tracer(tracer_index).asnumpy()
        )
        actual = outputs[name].asnumpy()
        test_utils.assert_dallclose(actual[:, jks:], reference[:, jks:], atol=1e-13)
        # above the cloudy region ICON does not run the scheme; the full-column
        # granule must produce (near-)zero tendencies there
        test_utils.assert_dallclose(actual[:, :jks], 0.0, atol=1e-12)

    tend_ta_reference = exit_savepoint.tend_ta().asnumpy() - init_savepoint.tend_ta().asnumpy()
    tend_ta_actual = outputs["tend_temperature"].asnumpy()
    test_utils.assert_dallclose(tend_ta_actual[:, jks:], tend_ta_reference[:, jks:], atol=1e-10)
    test_utils.assert_dallclose(tend_ta_actual[:, :jks], 0.0, atol=1e-10)

    # surface precip: the granule keeps the surface value in the last level; ICON
    # only stores the aggregated prm_field diagnostics (rsfl = rain,
    # ssfl = ice + snow + graupel, pr = total, ufcs = energy flux)
    rain = outputs["pr"].asnumpy()[:, -1]
    ice = outputs["pi"].asnumpy()[:, -1]
    snow = outputs["ps"].asnumpy()[:, -1]
    graupel = outputs["pg"].asnumpy()[:, -1]
    energy_flux = outputs["pre"].asnumpy()[:, -1]

    test_utils.assert_dallclose(rain, exit_savepoint.rsfl().asnumpy(), atol=1e-10)
    test_utils.assert_dallclose(ice + snow + graupel, exit_savepoint.ssfl().asnumpy(), atol=1e-10)
    test_utils.assert_dallclose(
        rain + ice + snow + graupel, exit_savepoint.pr().asnumpy(), atol=1e-10
    )
    test_utils.assert_dallclose(energy_flux, exit_savepoint.ufcs().asnumpy(), atol=1e-10)
