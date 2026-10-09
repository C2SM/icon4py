# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the prescribed surface fluxes.

exp.exclaim_ape_aesPhys runs with fixed kinematic surface heat fluxes (`isrfc_type = 1`) over
a constant SST, so the fluxes ICON feeds into the diffusion are a function of the surface
pressure alone; they are verified against the tmx-surface-fluxes savepoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import surface_fluxes, tmx_states
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import TMX_DATES, read_input_namelist


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_prescribed_surface_fluxes(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    testcase = read_input_namelist(experiment_description, process_props)["nh_testcase_nml"]
    assert testcase["ape_sst_case"] == "sst_const"
    allocator = model_backends.get_allocator(backend)
    reference = data_provider.from_savepoint_tmx_surface_fluxes(date=date)

    assert testcase["isrfc_type"] == 1
    provider = surface_fluxes.PrescribedFluxProvider(
        grid=icon_grid,
        backend=backend,
        surface_temperature=data_alloc.constant_field(
            icon_grid,
            constants.MELTING_TEMPERATURE + testcase["ape_sst_val"],
            dims.CellDim,
            allocator=allocator,
        ),
        # the defaults of `mo_nh_testcases_nml.f90` for the members the namelist leaves out
        shflx=testcase.get("shflx", 0.1),
        lhflx=testcase.get("lhflx", 0.0),
    )
    out = tmx_states.TmxSurfaceFluxState(
        evapotranspiration=data_alloc.zero_field(icon_grid, dims.CellDim, allocator=allocator),
        sensible_heat_flux=data_alloc.zero_field(icon_grid, dims.CellDim, allocator=allocator),
        u_stress=data_alloc.constant_field(icon_grid, 1.0, dims.CellDim, allocator=allocator),
        v_stress=data_alloc.constant_field(icon_grid, 1.0, dims.CellDim, allocator=allocator),
        q_snocpymlt=data_alloc.constant_field(icon_grid, 1.0, dims.CellDim, allocator=allocator),
    )
    provider.compute(
        pressure_ifc=data_provider.from_savepoint_tmx_entry(date=date).pres_ifc(), out=out
    )

    for name, computed, desired in (
        ("hfss", out.sensible_heat_flux, reference.hfss()),
        ("evspsbl", out.evapotranspiration, reference.evspsbl()),
        ("tauu", out.u_stress, reference.tauu()),
        ("tauv", out.v_stress, reference.tauv()),
        ("q_snocpymlt", out.q_snocpymlt, reference.q_snocpymlt()),
    ):
        test_utils.assert_dallclose(
            computed.asnumpy(), desired.asnumpy(), rtol=1.0e-14, err_msg=name
        )
    # the one field that is not zero in this configuration
    assert np.all(np.abs(out.sensible_heat_flux.asnumpy()) > 0.0)
