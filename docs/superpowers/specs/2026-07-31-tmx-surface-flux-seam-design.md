# TMX surface-flux provider seam (zero-flux implementation)

**Date:** 2026-07-31
**Status:** Approved design, pre-implementation
**Branch:** `physics_driver_tmx` (PR #1360)
**Predecessor:** `2026-07-07-tmx-component-integration-design.md` (phase-1; this design
resolves its "phase 2 sketch" *seam shape* decision, deliberately without the physics)

## Goal

Make the currently implicit surface-flux handling in `TmxState` an explicit, named seam:
a provider object that fills `TmxSurfaceFluxState` once per physics step. The first and
only implementation in this cycle writes zeros — pure plumbing, no physics. The real
ocean bulk-flux scheme (prescribed SST, Louis exchange coefficients, `z0m_oce`
roughness — `mo_tmx_surface.f90` / `mo_vdf_sfc.f90` / `mo_vdf_diag_smag.f90`) follows in
a later design, after the surface scheme is fully understood (JSBACH/land and sea-ice
tiles are out of scope for the aquaplanet: only the open-water tile is active,
`ape_sst_case='sst_const'`, SST = 303.15 K).

## Non-goals

- Any flux physics (bulk formulas, exchange coefficients, saturation humidity, SST
  handling). Behavior is exactly today's zero fluxes.
- A standalone `Component` for surface fluxes in the `PhysicsDriver`. Rejected for now:
  the driver's process model has no inter-process data handoff (gather → compute →
  scatter works on prognostic state only); extending it for a zero-writing provider is
  not warranted. Revisit only if radiation later forces a general mechanism.
- Fixing the provider's *input* contract. The seam fixes where and when fluxes are
  produced; the inputs the real scheme needs (lowest-level state, SST, roughness) are
  determined by the Fortran port and added to the `compute` signature then — a
  one-call-site change.

## Design

### Interface (new module `tmx/surface_fluxes.py`)

```python
class SurfaceFluxProvider(Protocol):
    def compute(self, *, out: tmx_states.TmxSurfaceFluxState) -> None: ...
```

- Called exactly once per physics step, as the final step of
  `TmxState.gather_from_prognostic` (after the thermodynamic diagnostics, before
  `Tmx.run` consumes the buffers via `as_component_input`).
- Must set every field of `out` on every call (no partial writes).

### Zero implementation (same module)

```python
class ZeroFluxProvider:
    def compute(self, *, out): ...  # writes 0.0 into all five buffers
```

- Explicitly **re-zeros** each call instead of no-op'ing: this upholds the
  "fluxes are set each step" contract even if the granule ever mutates the buffers in
  place. The buffers are 2-D (`CellField`), so the cost is negligible.
- `q_snocpymlt` is a land-tile (JSBACH) output and stays zero on the aquaplanet; the
  provider zeros it like the others.

### Wiring (`tmx/state.py`)

- `TmxState.__init__` gains a keyword arg
  `surface_flux_provider: SurfaceFluxProvider | None = None`; `None` resolves to
  `ZeroFluxProvider()`.
- `__init__` constructs one `TmxSurfaceFluxState` over its five existing buffers
  (frozen dataclass of references, built once, reused every step).
- `gather_from_prognostic` ends with
  `self._surface_flux_provider.compute(out=self._surface_flux_state)`.
- No driver or config changes: `driver_utils.initialize_granules` keeps constructing
  `TmxState` without the new argument, so the default preserves current behavior
  bit-for-bit.
- Cleanup in passing: the comment on the buffer allocations in `state.py` ("TMX fills
  them during a step; scatter reads them back to the land model") is wrong — the buffers
  are granule *inputs* filled from outside. Reword to describe the provider seam.
- Alignment in passing (audit vs the approved #1301 muphys pattern, 2026-07-31):
  `TmxState` explicitly inherits `PhysicsState` (`class TmxState(PhysicsState)`),
  matching muphys `State(PhysicsState)` — a colleague-approved review decision on #1301
  that lets mypy verify protocol conformance. All other integration aspects
  (PhysicsProcess wiring, ForcingMode.APPLY, timedelta convention, metadata vocabulary)
  were audited as already aligned; the `ddt_*` vs `tend_*` output naming difference is
  deliberate (each mirrors its own granule's port names) and stays.

### Data flow (unchanged path, new writer)

```
gather_from_prognostic:
  diagnostics (T, p, u/v, air_mass, cv_air)
  provider.compute(out=surface_flux_state)      # <- new explicit step
as_component_input:
  buffers -> input dict -> component packs TmxSurfaceFluxState -> Tmx.run
```

## Error handling

Nothing beyond the protocol contract: `compute` is infallible for the zero provider.
Future providers raise on their own invalid inputs; the seam adds no error paths.

## Testing (fast unit tests, simple grid, no data)

New `tmx/tests/tmx/unit_tests/test_surface_fluxes.py`:

1. Default construction: `TmxState` without the argument uses `ZeroFluxProvider`; after
   `gather_from_prognostic` all five buffers are zero.
2. Injection: a fake provider records calls and writes sentinel values; `gather` invokes
   it exactly once, and the sentinels appear in the `as_component_input` dict (same
   underlying buffers).
3. Re-zero: dirty the buffers, call `ZeroFluxProvider.compute`, assert all-zero again.

Existing TMX unit tests, the component datatests, and the APE_aes muphys+tmx smoke test
must pass unchanged (the change is behavior-preserving).

## New files

- `model/atmosphere/subgrid_scale_physics/tmx/src/icon4py/model/atmosphere/subgrid_scale_physics/tmx/surface_fluxes.py`
- `model/atmosphere/subgrid_scale_physics/tmx/tests/tmx/unit_tests/test_surface_fluxes.py`

## Follow-up (next design, after the colleague discussion)

Ocean bulk fluxes behind this seam: port the open-water branch of
`compute_sfc_roughness` → `sfc_exchange_coefficients` (Louis) →
`compute_sfc_sat_spec_humidity` → `compute_sfc_fluxes`, provider configured with the
prescribed SST; widen `SurfaceFluxProvider.compute` with the inputs that port needs and
validate against the `tmx-surface-fluxes` savepoints (v07 archive).
