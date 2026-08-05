"""Unit tests for the TMX surface-flux provider seam (simple grid, no data)."""

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import surface_fluxes, tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc

from .test_state import _tmx_state, _tracer_state, _uniform_prognostic


FLUX_NAMES = (
    "evapotranspiration",
    "sensible_heat_flux",
    "u_stress",
    "v_stress",
    "q_snocpymlt",
)


def _dirty_flux_state(grid) -> tmx_states.TmxSurfaceFluxState:
    """TmxSurfaceFluxState with distinct non-zero values in every field."""
    return tmx_states.TmxSurfaceFluxState(
        **{
            name: data_alloc.constant_field(grid, float(i + 1), dims.CellDim)
            for i, name in enumerate(FLUX_NAMES)
        }
    )


def test_zero_flux_provider_rezeros_all_fields():
    """compute() must set every flux field to zero, even if previously dirty."""
    grid = simple.simple_grid()
    out = _dirty_flux_state(grid)
    surface_fluxes.ZeroFluxProvider().compute(out=out)
    for name in FLUX_NAMES:
        np.testing.assert_array_equal(getattr(out, name).asnumpy(), 0.0, err_msg=name)


def test_gather_rezeros_fluxes_by_default():
    """Default TmxState (no provider arg) uses ZeroFluxProvider: gather re-zeros dirty buffers."""
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    state.sensible_heat_flux.ndarray[...] = 42.0  # dirty one buffer to prove re-zeroing
    state.gather_from_prognostic(_uniform_prognostic(grid), _tracer_state(grid, qv=1e-3))
    inp = state.as_component_input()
    for name in FLUX_NAMES:
        np.testing.assert_array_equal(inp[name].asnumpy(), 0.0, err_msg=name)


class _RecordingProvider:
    """Fake provider: counts calls and writes a sentinel into one buffer."""

    def __init__(self) -> None:
        self.calls = 0

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        self.calls += 1
        out.sensible_heat_flux.ndarray[...] = 123.0


def test_injected_provider_called_once_and_values_reach_component_input():
    grid = simple.simple_grid()
    provider = _RecordingProvider()
    state = _tmx_state(grid, surface_flux_provider=provider)
    state.gather_from_prognostic(_uniform_prognostic(grid), _tracer_state(grid, qv=1e-3))
    assert provider.calls == 1
    inp = state.as_component_input()
    np.testing.assert_array_equal(inp["sensible_heat_flux"].asnumpy(), 123.0)
