# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""TmxComponent: per-process adapter wrapping the Tmx turbulent-mixing granule."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.tmx import Tmx, TmxConfig, TmxParams
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.common.grid import base as base_grid


class TmxComponent:
    """The Tmx granule: a per-process adapter wrapping the AES turbulent mixing scheme.

    Wraps :class:`~icon4py.model.atmosphere.subgrid_scale_physics.tmx.tmx.Tmx`
    behind the Component protocol (``inputs_properties`` / ``outputs_properties``
    class attributes, ``__call__(state, time_step) -> dict``).

    Persistent output buffers (tendency and diagnostic states) are allocated once
    at construction time and reused across calls; the returned dict holds references
    to those buffers, not copies.
    """

    # TODO (Yilu): inherit the Component protocol once it is formalized
    # (deferred to a separate PR).

    inputs_properties = tmx_data.INPUTS_PROPERTIES
    outputs_properties = tmx_data.OUTPUTS_PROPERTIES

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: TmxConfig | None,
        metric_state: tmx_states.TmxMetricState | None,
        interpolation_state: tmx_states.TmxInterpolationState | None,
        edge_params: grid_states.EdgeParams | None,
        cell_params: grid_states.CellParams | None,
        dtime: datetime.timedelta,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None = None,
        exchange: decomposition.ExchangeRuntime = decomposition.single_node_exchange,
        granule: object | None = None,
    ) -> None:
        self._dt_seconds: float = dtime.total_seconds()

        allocator = model_backends.get_allocator(backend)

        if granule is None:
            # All physics state arguments must be provided when building the real granule.
            assert config is not None, "config must not be None when granule is not injected"
            assert metric_state is not None, (
                "metric_state must not be None when granule is not injected"
            )
            assert interpolation_state is not None, (
                "interpolation_state must not be None when granule is not injected"
            )
            assert edge_params is not None, (
                "edge_params must not be None when granule is not injected"
            )
            assert cell_params is not None, (
                "cell_params must not be None when granule is not injected"
            )
            granule = Tmx(
                grid=grid,
                config=config,
                params=TmxParams(config),
                vertical_grid=None,
                metric_state=metric_state,
                interpolation_state=interpolation_state,
                edge_params=edge_params,
                cell_params=cell_params,
                backend=backend,
                exchange=exchange,
            )

        self._granule = granule

        # Allocate persistent output buffers once; reused across every __call__.
        self._diagnostic_state: tmx_states.TmxDiagnosticState = (
            tmx_states.TmxDiagnosticState.allocate(grid, allocator=allocator)
        )
        self._tendency_state: tmx_states.TmxTendencyState = tmx_states.TmxTendencyState.allocate(
            grid, allocator=allocator
        )
        self._new_state: tmx_states.TmxNewState = tmx_states.TmxNewState.allocate(
            grid, allocator=allocator
        )

        # Pre-compute output field names from dataclass introspection so __call__
        # does not repeat the mapping logic.
        _tendency_names = {f.name for f in dataclasses.fields(tmx_states.TmxTendencyState)}
        _diagnostic_names = {f.name for f in dataclasses.fields(tmx_states.TmxDiagnosticState)}
        _output_keys = set(tmx_data.OUTPUTS_PROPERTIES)
        self._tendency_output_keys = _output_keys & _tendency_names
        self._diagnostic_output_keys = _output_keys & _diagnostic_names

    def __call__(
        self,
        state: dict,
        time_step: datetime.datetime,
    ) -> dict:
        """Run one tmx time step and return the output-contract dict.

        Builds :class:`~tmx_states.TmxInputState` and
        :class:`~tmx_states.TmxSurfaceFluxState` as frozen dataclasses of
        references from *state* (no copy), delegates to
        :meth:`~tmx.Tmx.run`, then returns a dict of
        :data:`~data.OUTPUTS_PROPERTIES` keys pointing to the persistent
        tendency and diagnostic buffers.

        Parameters
        ----------
        state:
            Mapping of field names → GT4Py fields; must contain all keys
            listed in :attr:`inputs_properties`.
        time_step:
            Wall-clock time of the current physics step (not currently used
            by Tmx, but part of the Component protocol).

        Returns
        -------
        dict
            Exactly the keys of :attr:`outputs_properties`, referencing the
            persistent internal buffers (no copies).
        """
        # Pack TmxInputState from references into state dict (no copy).
        input_state = tmx_states.TmxInputState(
            **{f.name: state[f.name] for f in dataclasses.fields(tmx_states.TmxInputState)}
        )

        # Pack TmxSurfaceFluxState from references into state dict (no copy).
        surface_flux_state = tmx_states.TmxSurfaceFluxState(
            **{f.name: state[f.name] for f in dataclasses.fields(tmx_states.TmxSurfaceFluxState)}
        )

        self._granule.run(
            input_state=input_state,
            surface_flux_state=surface_flux_state,
            diagnostic_state=self._diagnostic_state,
            tendency_state=self._tendency_state,
            new_state=self._new_state,
            dtime=self._dt_seconds,
        )

        return {
            **{k: getattr(self._tendency_state, k) for k in self._tendency_output_keys},
            **{k: getattr(self._diagnostic_state, k) for k in self._diagnostic_output_keys},
        }
