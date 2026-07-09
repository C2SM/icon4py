# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Component-level datatest for TmxComponent against tmx entry/exit savepoints.

Verifies that ``TmxComponent`` driven through the dict interface reproduces the
tmx-exit savepoint, and that a second consecutive call does not leak through the
reused output buffers.  A second test validates the gather-thermodynamics layer
(``compute_air_mass`` / ``compute_cv_air`` programs) against the entry savepoint.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    state_stencils,
)
from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    assert_scaled_allclose,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
    verify_full_run_fields,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_component_reproduces_exit_savepoint(
    *,
    date: str,
    icon_grid: icon_grid_.IconGrid,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    data_provider: sb.IconSerialDataProvider,
    tmx_config,
    tmx_dtime: float,
    backend: gtx_typing.Backend | None,
) -> None:
    """TmxComponent(dict interface) must reproduce the Fortran tmx-exit savepoint.

    The granule is constructed from the same savepoint-derived states as
    ``test_tmx_full_run_single_step``; inputs are routed through the Component's
    dict contract instead of the raw ``Tmx.run`` call.  A second identical call
    is used to confirm that the persistent output buffers are cleanly overwritten
    on reuse (no state leak between steps).
    """
    allocator = model_backends.get_allocator(backend)
    # init_savepoint is not a pytest fixture; loaded from data_provider as in test_tmx_run.py
    init_savepoint = data_provider.from_savepoint_tmx_init()

    comp = tmx_component.TmxComponent(
        grid=icon_grid,
        config=tmx_config,
        metric_state=construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=init_savepoint,
            grid_savepoint=grid_savepoint,
            allocator=allocator,
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        dtime=datetime.timedelta(seconds=tmx_dtime),
        backend=backend,
    )

    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    surface_fluxes_savepoint = data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    exit_savepoint = data_provider.from_savepoint_tmx_exit(date=date)

    input_state = construct_input_state(entry_savepoint)
    flux_state = construct_surface_flux_state(surface_fluxes_savepoint)
    state_dict = {
        **{f.name: getattr(input_state, f.name) for f in dataclasses.fields(input_state)},
        **{f.name: getattr(flux_state, f.name) for f in dataclasses.fields(flux_state)},
    }

    # Strip trailing zeros then trailing dot so fromisoformat works on Python ≤3.10
    time_step = datetime.datetime.fromisoformat(date.rstrip("0").rstrip("."))
    outputs = comp(state_dict, time_step)

    verify_full_run_fields(
        diagnostic_state=comp._diagnostic_state,
        tendency_state=comp._tendency_state,
        exit_savepoint=exit_savepoint,
        num_levels=icon_grid.num_levels,
    )
    # The dict view must reference the persistent buffers directly (no copies)
    assert outputs["ddt_temperature"] is comp._tendency_state.ddt_temperature

    # Second call with the same inputs: verify the reused buffers are cleanly overwritten
    outputs = comp(state_dict, time_step)
    verify_full_run_fields(
        diagnostic_state=comp._diagnostic_state,
        tendency_state=comp._tendency_state,
        exit_savepoint=exit_savepoint,
        num_levels=icon_grid.num_levels,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_gather_air_mass_and_cv_air_match_savepoint(
    *,
    date: str,
    icon_grid: icon_grid_.IconGrid,
    metrics_savepoint: sb.MetricSavepoint,
    data_provider: sb.IconSerialDataProvider,
    backend: gtx_typing.Backend | None,
) -> None:
    """compute_air_mass / compute_cv_air programs must reproduce the Fortran mair/cvair fields.

    The entry savepoint carries both the raw inputs (rho, q*) and the Fortran-computed
    derived fields (mair, cvair), so the stencils can be verified against real ICON data.
    """
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)

    air_mass = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    cv_air = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)

    full_horizontal = {
        "horizontal_start": gtx.int32(0),
        "horizontal_end": gtx.int32(icon_grid.num_cells),
    }
    full_vertical = {
        "vertical_start": gtx.int32(0),
        "vertical_end": gtx.int32(icon_grid.num_levels),
    }

    run_air_mass = model_options.setup_program(
        program=state_stencils.compute_air_mass,
        backend=backend,
        horizontal_sizes=full_horizontal,
        vertical_sizes=full_vertical,
        offset_provider={},
    )
    run_cv_air = model_options.setup_program(
        program=state_stencils.compute_cv_air,
        backend=backend,
        horizontal_sizes=full_horizontal,
        vertical_sizes=full_vertical,
        offset_provider={},
    )

    run_air_mass(
        rho=entry_savepoint.rho(),
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
        air_mass=air_mass,
    )
    run_cv_air(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qi=entry_savepoint.qi(),
        qr=entry_savepoint.qr(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
        air_mass=air_mass,
        cv_air=cv_air,
    )

    assert_scaled_allclose(air_mass.asnumpy(), entry_savepoint.mair().asnumpy(), err_msg="mair")
    assert_scaled_allclose(cv_air.asnumpy(), entry_savepoint.cvair().asnumpy(), err_msg="cvair")
