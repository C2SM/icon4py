# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test of the tmx component.

Runs one step of `TmxComponent` from the tmx-entry savepoint, with the serialized surface
fluxes, and verifies it against the tmx-exit savepoint. Unlike the granule's test
(test_tmx.py), the air mass and the heat capacity of the air are derived by the component,
not read from the savepoint.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    component as tmx_component,
    tmx_states,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.components import framework as fw, quantities as qty
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .utils import (
    TMX_DATES,
    assert_tmx_exit_fields,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
)


if TYPE_CHECKING:
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta
    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


@dataclasses.dataclass(frozen=True)
class SerializedFluxProvider:
    """The surface fluxes ICON computed for the step, as serialized."""

    fluxes: tmx_states.TmxSurfaceFluxState

    def compute(
        self,
        *,
        pressure_ifc: fa.CellKHalfField[ta.wpfloat],
        out: tmx_states.TmxSurfaceFluxState,
    ) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = getattr(self.fluxes, field.name).ndarray


def _inputs(entry: sb.TmxEntrySavepoint) -> tmx_component.TmxComponent.Input:
    return tmx_component.TmxComponent.Input(
        temperature=fw.Field(qty.TemperatureOnCellK, entry.ta()),
        virtual_temperature=fw.Field(qty.VirtualTemperatureOnCellK, entry.tempv()),
        pressure=fw.Field(qty.PressureOnCellK, entry.pres()),
        pressure_ifc=fw.Field(qty.PressureOnCellKHalf, entry.pres_ifc()),
        u=fw.Field(qty.UOnCellK, entry.ua()),
        v=fw.Field(qty.VOnCellK, entry.va()),
        w=fw.Field(qty.WOnCellKHalf, entry.wa()),
        rho=fw.Field(qty.RhoOnCellK, entry.rho()),
        qv=fw.Field(qty.QvOnCellK, entry.qv()),
        qc=fw.Field(qty.QcOnCellK, entry.qc()),
        qi=fw.Field(qty.QiOnCellK, entry.qi()),
        qr=fw.Field(qty.QrOnCellK, entry.qr()),
        qs=fw.Field(qty.QsOnCellK, entry.qs()),
        qg=fw.Field(qty.QgOnCellK, entry.qg()),
    )


def _leaves(out: tmx_component.TmxComponent.Output, state: type[Any]) -> dict[str, Any]:
    """The Output leaves that make up a tmx state class, by name."""
    return {field.name: getattr(out, field.name).data for field in dataclasses.fields(state)}


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    [(definitions.Experiments.EXCLAIM_APE_AES, date) for date in TMX_DATES],
)
def test_tmx_component_run_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    decomposition_info: decomposition.DecompositionInfo,
    backend: model_backends.BackendLike,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    tmx_config = experiment.config.tmx
    assert tmx_config is not None
    component = tmx_component.TmxComponent(
        grid=icon_grid,
        config=tmx_config,
        dtime=experiment.config.driver.dtime,
        metric_state=construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=data_provider.from_savepoint_tmx_init(),
            allocator=model_backends.get_allocator(backend),
        ),
        interpolation_state=construct_interpolation_state(interpolation_savepoint),
        edge_params=grid_savepoint.construct_edge_geometry(),
        cell_params=grid_savepoint.construct_cell_geometry(),
        surface_fluxes=SerializedFluxProvider(
            construct_surface_flux_state(data_provider.from_savepoint_tmx_surface_fluxes(date=date))
        ),
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
    )

    out = component.run(_inputs(data_provider.from_savepoint_tmx_entry(date=date)))

    assert_tmx_exit_fields(
        tendency_state=tmx_states.TmxTendencyState(**_leaves(out, tmx_states.TmxTendencyState)),
        diagnostic_state=tmx_states.TmxDiagnosticState(
            **_leaves(out, tmx_states.TmxDiagnosticState)
        ),
        exit_savepoint=data_provider.from_savepoint_tmx_exit(date=date),
        use_km_const=tmx_config.use_km_const,
        owner_mask=data_alloc.as_numpy(decomposition_info.owner_mask(dims.CellDim)),
    )
