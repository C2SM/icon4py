# TMX Surface-Flux Provider Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TMX's surface-flux handling an explicit provider seam (`SurfaceFluxProvider` protocol + `ZeroFluxProvider`) called once per physics step from `TmxState.gather_from_prognostic`, plus align `TmxState` with the approved `PhysicsState` inheritance pattern.

**Architecture:** New module `tmx/surface_fluxes.py` holds the protocol and the zero implementation; `TmxState` takes a provider at construction (default zero) and invokes it as the final gather step, writing into the five existing 2-D flux buffers via a `TmxSurfaceFluxState` view built once. Behavior is bit-for-bit unchanged (fluxes stay zero); no driver or config changes.

**Tech Stack:** Python 3.12, GT4Py fields (`.ndarray` in-place writes), pytest on `simple_grid` (no serialized data), uv workspace.

**Spec:** `docs/superpowers/specs/2026-07-31-tmx-surface-flux-seam-design.md`

## Global Constraints

- Run everything from the repo root: `/Users/chenyilu/Desktop/01_Work/EXCLAIM/icon4py/.worktrees/physics_driver_tmx`
- Test command prefix: `uv run --group test --frozen pytest -q -n0 --benchmark-disable`
- Do NOT add license headers to new files by hand — the `insert-license` pre-commit hook adds them from `HEADER.txt`.
- **Yilu commits manually.** Commit steps below are checkpoints: stop, summarize the change, and let Yilu review + commit. Never run `git commit`.
- Behavior must be preserved exactly: all existing TMX and standalone_driver unit tests pass unchanged.
- mypy note: `tmx/src` is deliberately NOT in the pyproject `[tool.mypy].files` list (commented out), so pre-commit mypy will not check the new code; do not add it to the list (out of scope — the upstream `tmx.py` is not mypy-clean).

---

### Task 1: Align `TmxState` with the `PhysicsState` protocol

**Files:**
- Modify: `model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state.py` (imports ~line 16, class statement line 67, module docstring lines 9-15)

**Interfaces:**
- Consumes: `PhysicsState` protocol from `icon4py.model.common.components.physics_state` (methods: `gather_from_prognostic(prognostic, tracers)`, `as_component_input()`, `scatter_to_prognostic(prognostic, outputs, dtime)` — `TmxState` already implements all three).
- Produces: `class TmxState(PhysicsState)` — Tasks 2-3 modify the same file and must keep this base class.

- [ ] **Step 1: Add the explicit inheritance (mirrors muphys `State(PhysicsState)`)**

In `state.py`, add to the runtime imports (after the `from icon4py.model.common import (...)` block, matching muphys `state.py:26`):

```python
from icon4py.model.common.components.physics_state import PhysicsState
```

Change the class statement:

```python
class TmxState(PhysicsState):
```

- [ ] **Step 2: Fix the stale module docstring**

The module docstring ends with a sentence that predates the scatter implementation. Replace:

```
Only the gather half (plus ``as_component_input``) is
implemented here; ``scatter_to_prognostic`` is deferred to Task 5.
```

with:

```
All three protocol methods (``gather_from_prognostic``,
``as_component_input``, ``scatter_to_prognostic``) are implemented here.
```

- [ ] **Step 3: Run the TMX unit tests to verify nothing changed**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/`
Expected: all pass (same count as before the change; currently the suite is green).

- [ ] **Step 4: Checkpoint — Yilu reviews and commits**

Suggested message: `refactor(tmx): TmxState explicitly inherits PhysicsState (align with muphys pattern)`

---

### Task 2: `surface_fluxes.py` — provider protocol + `ZeroFluxProvider`

**Files:**
- Create: `model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/surface_fluxes.py`
- Create: `model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py`

**Interfaces:**
- Consumes: `tmx_states.TmxSurfaceFluxState` (frozen dataclass with fields `evapotranspiration`, `sensible_heat_flux`, `u_stress`, `v_stress`, `q_snocpymlt`, all `fa.CellField[ta.wpfloat]`).
- Produces: `SurfaceFluxProvider` protocol with `compute(*, out: tmx_states.TmxSurfaceFluxState) -> None`, and `ZeroFluxProvider` implementing it. Task 3 imports both.

- [ ] **Step 1: Write the failing test**

Create `test_surface_fluxes.py`:

```python
"""Unit tests for the TMX surface-flux provider seam (simple grid, no data)."""

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import surface_fluxes, tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.utils import data_allocation as data_alloc


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py -v`
Expected: FAIL at import with `ImportError: cannot import name 'surface_fluxes'` (module does not exist yet).

- [ ] **Step 3: Write the module**

Create `surface_fluxes.py`:

```python
"""Surface-flux provider seam for TMX.

The provider fills the surface-flux input buffers of the granule
(:class:`~icon4py.model.atmosphere.subgrid_scale_physics.tmx.tmx_states.TmxSurfaceFluxState`)
once per physics step. This cycle ships only :class:`ZeroFluxProvider` (pure
plumbing); the ocean bulk-flux scheme (prescribed SST, Louis exchange
coefficients — ``mo_tmx_surface.f90`` / ``mo_vdf_sfc.f90``) is a follow-up
implementation behind the same seam. Design:
``docs/superpowers/specs/2026-07-31-tmx-surface-flux-seam-design.md``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states


class SurfaceFluxProvider(Protocol):
    """Fills TMX's surface-flux input buffers, once per physics step."""

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        """Set every field of ``out``.

        Called as the final step of ``TmxState.gather_from_prognostic`` (after
        the thermodynamic diagnostics, before ``Tmx.run`` consumes the
        buffers). Implementations must write all fields on every call — no
        partial updates.
        """
        ...


class ZeroFluxProvider:
    """Zero surface fluxes (the phase-2 seam's plumbing-only implementation).

    Explicitly re-zeros every call instead of no-op'ing: this upholds the
    "fluxes are set each step" contract even if the granule ever mutated the
    buffers in place. The buffers are 2-D, so the cost is negligible.
    """

    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None:
        for field in dataclasses.fields(out):
            getattr(out, field.name).ndarray[...] = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py -v`
Expected: `test_zero_flux_provider_rezeros_all_fields PASSED`

- [ ] **Step 5: Checkpoint — Yilu reviews and commits**

Suggested message: `feat(tmx): SurfaceFluxProvider seam — protocol + ZeroFluxProvider`

---

### Task 3: Wire the provider into `TmxState.gather_from_prognostic`

**Files:**
- Modify: `model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/state.py` (imports, `__init__` signature + body around the flux-buffer allocations ~line 240, end of `gather_from_prognostic` ~line 342)
- Modify: `model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_state.py` (`_tmx_state` helper, ~line 65: add `**kwargs` passthrough)
- Test: `model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py` (extend)

**Interfaces:**
- Consumes: `surface_fluxes.SurfaceFluxProvider` / `surface_fluxes.ZeroFluxProvider` (Task 2); `tmx_states.TmxSurfaceFluxState`; `TmxState(PhysicsState)` (Task 1).
- Produces: `TmxState.__init__(..., surface_flux_provider: surface_fluxes.SurfaceFluxProvider | None = None)`; gather calls `provider.compute(out=...)` exactly once per invocation. Call sites without the argument (driver_utils, existing tests) keep zero fluxes.

- [ ] **Step 1: Extend the `_tmx_state` helper to forward kwargs**

In `test_state.py`, change the helper signature and construction:

```python
def _tmx_state(grid, **kwargs) -> TmxState:
    """Construct a TmxState on the simple grid with neutral/zero interpolation coefficients."""
    return TmxState(
        grid=grid,
        ddqz_z_full=data_alloc.constant_field(grid, 100.0, dims.CellDim, dims.KDim),
        rbf_coeff_c1=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        rbf_coeff_c2=data_alloc.zero_field(grid, dims.CellDim, dims.C2E2C2EDim),
        c_lin_e=data_alloc.constant_field(grid, 0.5, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_x=data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.E2CDim),
        primal_normal_cell_y=data_alloc.zero_field(grid, dims.EdgeDim, dims.E2CDim),
        backend=None,
        **kwargs,
    )
```

- [ ] **Step 2: Write the two failing tests**

Append to `test_surface_fluxes.py`:

```python
from .test_state import _tmx_state, _tracer_state, _uniform_prognostic


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py -v`
Expected: `test_gather_rezeros_fluxes_by_default` FAILS on `sensible_heat_flux` (the test dirties that buffer and nothing re-zeros it yet); `test_injected_provider_called_once_...` FAILS with `TypeError: __init__() got an unexpected keyword argument 'surface_flux_provider'`.

- [ ] **Step 4: Implement the wiring in `state.py`**

Add to the imports (with the existing `from icon4py.model.atmosphere...tmx import state_stencils` block):

```python
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import (
    state_stencils,
    surface_fluxes,
    tmx_states,
)
```

Extend the `__init__` signature (after `primal_normal_cell_y`, before `backend`):

```python
        surface_flux_provider: surface_fluxes.SurfaceFluxProvider | None = None,
```

Replace the flux-buffer comment and add the provider + view after the five buffer allocations (the spec's comment fix — the buffers are granule *inputs* filled by the provider, not outputs read back to a land model):

```python
        # --- Surface-flux buffers: 2-D (CellDim only), granule *inputs*.
        #     The surface-flux provider (phase-2 seam) fills them at the end of
        #     every gather; the granule consumes them via as_component_input.
        self.evapotranspiration = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.sensible_heat_flux = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.u_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.v_stress = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self.q_snocpymlt = data_alloc.zero_field(grid, dims.CellDim, allocator=backend)
        self._surface_flux_provider = surface_flux_provider or surface_fluxes.ZeroFluxProvider()
        self._surface_flux_state = tmx_states.TmxSurfaceFluxState(
            evapotranspiration=self.evapotranspiration,
            sensible_heat_flux=self.sensible_heat_flux,
            u_stress=self.u_stress,
            v_stress=self.v_stress,
            q_snocpymlt=self.q_snocpymlt,
        )
```

Append as the final step of `gather_from_prognostic` (after the `# 5. cv_air` block):

```python
        # 6. Surface fluxes (phase-2 seam): the provider sets every flux, every step
        self._surface_flux_provider.compute(out=self._surface_flux_state)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Run the full TMX unit suite (regression)**

Run: `uv run --group test --frozen pytest -q -n0 --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/`
Expected: all pass (previous suite + 3 new).

- [ ] **Step 7: Checkpoint — Yilu reviews and commits**

Suggested message: `feat(tmx): TmxState fills surface fluxes via injected provider each gather`

---

### Task 4: Full verification sweep

**Files:** none (verification only)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: green suite + clean pre-commit as the merge-readiness evidence.

- [ ] **Step 1: Run the affected unit suites**

Run: `uv run --group test --frozen pytest -q -n0 --datatest-skip --benchmark-disable model/atmosphere/subgrid_scale_physics/tmx/tests/ model/atmosphere/subgrid_scale_physics/physics_driver/tests/ model/standalone_driver/tests/`
Expected: all pass, none skipped unexpectedly (datatests deselected — they need the v07 archive).

- [ ] **Step 2: Run pre-commit on all files**

Run: `uv run --group dev --frozen --isolated pre-commit run --all-files`
Expected: every hook passes (the `insert-license` hook adds headers to the two new files on first run — if it modifies them, re-run to confirm it then passes).

- [ ] **Step 3: Checkpoint — Yilu reviews the diff and commits anything the hooks fixed**

Suggested message (if hooks changed files): `style(tmx): pre-commit formatting for the surface-flux seam`

---

## Self-review notes

- Spec coverage: interface → Task 2; zero implementation (explicit re-zero) → Task 2; wiring incl. default + view built once + gather call → Task 3; comment cleanup → Task 3 Step 4; `PhysicsState` alignment → Task 1; testing section's three tests → Tasks 2-3; "no driver/config changes" → no task touches them (verified by Task 4 driver suite).
- Types consistent: `compute(*, out: tmx_states.TmxSurfaceFluxState) -> None` identical in protocol, implementation, fake, and call site.
- No placeholders; every code step shows the code.
