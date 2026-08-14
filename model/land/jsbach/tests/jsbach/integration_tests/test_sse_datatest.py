# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the ported soil-energy kernels against ICON's own savepoints.

ICON writes `sse-solve-entry` / `sse-solve-exit` around JSBACH's soil temperature
solve (mo_icon4py_verification.f90:1765-1860, called from tmx/mo_tmx_surface.f90).
This test replays one such step with the ported GT4Py kernels, in the Fortran's
order -- back substitution with the OLD coefficients, forward elimination on the
new temperatures, then the ground heat flux -- and compares against the exit state.

Preconditions baked into the experiment (see the runscript's header): l_freeze is
off, so no freeze/melt mutates the soil temperature between the two halves of the
solve, and the comparison is restricted to snow-free columns, where the surface
quantities handed back to the surface energy balance are pure soil.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.land.jsbach import soil_thermal_properties as thermal
from icon4py.model.land.jsbach.stencils import soil_temperature as soil
from icon4py.model.testing import definitions as test_defs, test_utils

from ..fixtures import *  # noqa: F403  [pytest resolves fixtures from this namespace]


#: The steps ICON serialized in this experiment, minus the first. On the lstart step
#: JSBACH skips the two-time-level back substitution (the soil state has no previous
#: coefficients yet), so the entry/exit pair at 00:00:00 is not a solve to compare
#: against -- the other kernels match there, but t_soil_sl does not.
SSE_DATES = [
    "2008-09-01T00:05:00.000",
    "2008-09-01T00:10:00.000",
]

#: ICON is built with nvfortran and `-acc=gpu`, which contracts `a + b*c` into a fused
#: multiply-add; so does the C++ compiler behind the gtfn backends, which is why
#: gtfn_cpu reproduces ICON bit-for-bit. The embedded backend evaluates through numpy
#: and does not fuse, so it lands a few ulp away. These tolerances admit that and
#: nothing more -- a genuine port error moves these fields by far more.
#: See model/land/jsbach/docs/fma_contraction.md.
RTOL = 1.0e-13
#: grnd_hflx is the one field where a single unfused operation is visible: it is a
#: difference of two O(300 K) terms scaled to O(1 W/m^2), so the cancellation lifts a
#: last-bit difference to ~1e-10 relative. Gate it on an absolute tolerance instead.
GRND_HFLX_ATOL = 1.0e-9


def _zeros(num_cells: int, num_levels: int, backend) -> gtx.Field:
    return gtx.zeros(
        {dims.CellDim: num_cells, dims.KDim: num_levels}, dtype=float, allocator=backend
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.EXCLAIM_APE_AES])
@pytest.mark.parametrize("date", SSE_DATES)
def test_soil_temperature_solve_matches_icon(
    date: str,
    *,
    data_provider,
    backend,
) -> None:
    geometry = data_provider.from_sse_geometry_savepoint()
    entry = data_provider.from_sse_solve_entry_savepoint(date=date)
    exit_ = data_provider.from_sse_solve_exit_savepoint(date=date)

    vertical_grid = thermal.soil_thermal_grid(geometry.soil_bots().asnumpy())
    num_levels = geometry.nsoil()
    delta_time = entry.dtime()

    acoef_old, bcoef_old = entry.t_soil_acoef(), entry.t_soil_bcoef()
    t_soil_new = exit_.t_soil_sl()
    vol_heat_cap, heat_cond = exit_.vol_heat_cap_sl(), exit_.heat_cond_sl()
    num_cells = t_soil_new.shape[0]

    # The kernels are the pure-soil, snow-free slice of the solve: on snow-covered
    # columns the top boundary and the returned surface quantities are blends of snow
    # and soil, and match no single kernel.
    snow_free = entry.snow_depth_sl().asnumpy().max(axis=1) == 0.0
    assert snow_free.any(), "no snow-free columns to compare"
    domain = dict(
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(num_levels),
    )

    # 1. back substitution, driven by the previous step's coefficients and the surface
    #    temperature (which is the soil column's top boundary where there is no snow)
    t_soil = _zeros(num_cells, num_levels, backend)
    soil.soil_temperature_back_substitution.with_backend(backend)(
        t_soil_acoef=acoef_old,
        t_soil_bcoef=bcoef_old,
        t_soil_top=exit_.t_srf(),
        t_soil_sl=t_soil,
        offset_provider={},
        **domain,
    )
    assert test_utils.dallclose(
        t_soil.asnumpy()[snow_free], t_soil_new.asnumpy()[snow_free], rtol=RTOL
    )

    # 2. forward elimination: the coefficients carried into the next step
    acoef, bcoef = _zeros(num_cells, num_levels, backend), _zeros(num_cells, num_levels, backend)
    soil.soil_temperature_coefficients.with_backend(backend)(
        t_soil_sl=t_soil_new,
        vol_heat_cap=vol_heat_cap,
        heat_cond=heat_cond,
        dz=vertical_grid.dz,
        zd1=vertical_grid.zd1,
        delta_time=delta_time,
        t_soil_acoef=acoef,
        t_soil_bcoef=bcoef,
        offset_provider={},
        **domain,
    )
    assert test_utils.dallclose(
        acoef.asnumpy()[snow_free], exit_.t_soil_acoef().asnumpy()[snow_free], rtol=RTOL
    )
    assert test_utils.dallclose(
        bcoef.asnumpy()[snow_free], exit_.t_soil_bcoef().asnumpy()[snow_free], rtol=RTOL
    )

    # 3. the surface quantities handed back to the surface energy balance, at the
    #    ground level (the kernel evaluates them per level; only k=0 is the surface)
    grnd_hflx = _zeros(num_cells, num_levels, backend)
    hcap_grnd = _zeros(num_cells, num_levels, backend)
    soil.soil_ground_heat_flux.with_backend(backend)(
        t_soil_sl=t_soil_new,
        t_soil_acoef=acoef,
        t_soil_bcoef=bcoef,
        vol_heat_cap=vol_heat_cap,
        heat_cond=heat_cond,
        dz=vertical_grid.dz,
        zd1=vertical_grid.zd1,
        delta_time=delta_time,
        grnd_hflx=grnd_hflx,
        hcap_grnd=hcap_grnd,
        offset_provider={},
        **domain,
    )
    assert test_utils.dallclose(
        grnd_hflx.asnumpy()[snow_free, 0],
        exit_.grnd_hflx().asnumpy()[snow_free],
        rtol=RTOL,
        atol=GRND_HFLX_ATOL,
    )
    assert test_utils.dallclose(
        hcap_grnd.asnumpy()[snow_free, 0], exit_.hcap_grnd().asnumpy()[snow_free], rtol=RTOL
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.EXCLAIM_APE_AES])
def test_soil_thermal_grid_matches_icon(*, data_provider) -> None:
    """The host-side vertical geometry must reproduce ICON's soil_depth_energy grid."""
    geometry = data_provider.from_sse_geometry_savepoint()
    vertical_grid = thermal.soil_thermal_grid(geometry.soil_bots().asnumpy())

    assert test_utils.dallclose(vertical_grid.dz.asnumpy(), geometry.soil_dz().asnumpy())
    assert test_utils.dallclose(vertical_grid.mids.asnumpy(), geometry.soil_mids().asnumpy())
    assert test_utils.dallclose(vertical_grid.bots.asnumpy(), geometry.soil_bots().asnumpy())
    # zd1 is not serialized: it is the inverse spacing between mid-depths, with the
    # bottom entry unused (mo_sse_process.f90:437).
    mids = geometry.soil_mids().asnumpy()
    assert test_utils.dallclose(vertical_grid.zd1.asnumpy()[:-1], 1.0 / np.diff(mids))
