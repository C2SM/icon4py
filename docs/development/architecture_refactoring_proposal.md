# ICON4Py Layered Architecture: Analysis and Refactoring Proposal

Status: **proposal** (analysis verified against the codebase as of 2026-07; refactoring pending)

## Context

ICON4Py is a GT4Py-based Python port of the Fortran ICON weather model: a uv monorepo of 11
namespace packages (~65k LOC src, ~50k LOC tests). Today it reads as "a port of ICON granules
plus transitional scaffolding": a 21k-LOC god package (`common`), two drivers, two microphysics
packages with incompatible styles, a dead uniform-component abstraction, a hand-rolled lazy-DAG
DI engine for static fields, and a `testing` package whose declared dependencies are false.

This proposal turns ICON4Py into a **modular, lightweight library of pluggable weather-model
components** that compose into applications — the full ICON configuration being just one
composition — aligned with the modern Python/ML-weather ecosystem (pace/NDSL, Sympl,
CliMA, xarray/CF). The target architecture defines five layers automatically enforced by
`tach` (already pinned `>=0.23.0`, so `layers` and `[[interfaces]]` are available). ICON
Fortran interfaces and configuration conventions are NOT preserved.

**Scope decisions:**

1. Namespace: keep an `icon4py.<earth-subsystem>.*` prefix for model components (e.g.
   `icon4py.atmosphere.*`; future `icon4py.ocean.*`, `icon4py.land.*`); drop `icon4py.model.`.
2. Fortran stack: **spin out py2fgen** to its own repo/distribution; **keep bindings** in-repo
   as a quarantined top-layer adapter. Serialbox validation datatests are kept throughout —
   they are the numerical safety net.
3. Packaging: consolidate 11 distributions to ~6; tach enforces internal boundaries.

## Current Architecture Analysis (verified)

### Package inventory & declared dependency graph (tach.toml today)

| Package                                             | LOC (src) | Depends on (tach)                                                                     |
| --------------------------------------------------- | --------: | ------------------------------------------------------------------------------------- |
| model/common                                        |     21.4k | — (foundation)                                                                        |
| model/atmosphere/dycore                             |     10.5k | common                                                                                |
| model/atmosphere/advection                          |      8.0k | common                                                                                |
| model/atmosphere/muphys                             |      4.2k | common                                                                                |
| model/atmosphere/subgrid_scale_physics/microphysics |      3.9k | common                                                                                |
| model/atmosphere/diffusion                          |      2.9k | common                                                                                |
| model/standalone_driver                             |      3.3k | common (understated: really dycore+diffusion+advection+microphysics)                  |
| model/driver (legacy)                               |      2.5k | diffusion, dycore, common (+testing)                                                  |
| model/testing                                       |      4.9k | common (**false**: imports advection/diffusion/dycore/microphysics/standalone_driver) |
| tools (py2fgen)                                     |      1.6k | —                                                                                     |
| bindings                                            |      1.9k | diffusion, dycore, muphys, common, tools.py2fgen                                      |

### Principal problems

01. **`common` is a god package** — grid (232KB), interpolation (133KB), metrics (122KB),
    decomposition (68KB), states (54KB), io, math, topography, diagnostic_calculations, config,
    components, utils + ~10 top-level modules. Also: grid↔decomposition module cycle
    (`decomposition/halo.py` imports `grid.base`+`gridfile`; `grid/grid_manager.py` imports
    `decomposition.definitions`).
02. **Overengineered static-field factory** (`common/states/factory.py`, 31KB): `FieldSource`
    protocol registry + 4 `FieldProvider` variants + `NeedsExchange` mixin + `CompositeSource`/
    ChainMap; runtime type-hint reflection for dependency validation; duplicated
    `_get_offset_providers`; `replace_khalfdim` workaround ×3; mutable-default `_providers`
    dict on the Protocol class. Powers geometry → interpolation → metrics field computation.
03. **Horizontal/vertical grid duplication & entanglement**: two incompatible `Zone`/`Domain`/
    `domain()` triplets in `grid/horizontal.py` vs `grid/vertical.py` (papered over by
    `factory.DomainType` TypeVar); vertical size embedded in horizontal `GridConfig` (TODOs
    acknowledge).
04. **Three parallel Fortran-config mechanisms**: `config/options.py` (Annotated-based, unused),
    `utils/fortran_config.config_dataclass_from_dict`, and hand-written `from_fortran_dict`
    classmethods on every config class.
05. **Dead uniform-component abstraction**: `common/components/components.py` `Component`
    protocol (Sympl-style, matches accepted ADR 0001 "physics returns tendencies") implemented
    by nothing (verified: only `components.monitor` is imported, by `io/io.py`). Every component
    has a bespoke run signature: `SolveNonhydro.time_step(...)`, `Diffusion.run(initial_run=...)`,
    `Advection.run(...)`, graupel `run(dtime, qv..qg kwargs)`, muphys = bare `graupel_run` program.
06. **Two microphysics packages**, different lineages/styles: `microphysics` (ICON one-moment
    six-class graupel; stateful-granule OO) vs `muphys` (Stevens scheme; functional GT4Py +
    655-line `graupel_dace_hooks.py` + own standalone NetCDF drivers). No shared interface.
07. **Two drivers with copy-pasted time-stepping** (`_integrate_one_time_step`,
    `_do_dyn_substepping`, …): legacy `model/driver` (click, hardcoded config, serialbox-only)
    vs `model/standalone_driver` (typer, namelist-JSON config, optional granules, NetCDF IO).
    Both depend on `icon4py-testing` (TODO: remove). standalone_driver imports advection +
    microphysics config **without declaring them in pyproject** (works only via workspace sync).
08. **Per-component private state duplication**: dycore/diffusion/advection each define their own
    `InterpolationState`/`MetricState`; `driver_utils.initialize_granules` hand-maps factory
    outputs into them (~175 lines); dycore `PrepAdvection` ≈ advection `AdvectionPrepAdvState`.
09. **Granule boilerplate copy-pasted per component**: `__init__` + `_determine_local_domains` +
    eager `setup_program` binding + `_allocate_local_fields`.
10. **`testing` mixes generic test infra with ICON-reference validation** (79KB `serialbox.py`,
    ~30 savepoint classes, experiment registry importing ALL components); `ExperimentConfig`
    duplicated in `standalone_driver/config.py` "to avoid circular imports".
11. **Global mutable state**: `type_alias.set_precision()` mutates module globals wpfloat/vpfloat;
    module-level `single_node_exchange`/`single_node_reductions` singletons as default args.
12. **`io` inside common** with a bloated extra (uxarray==2024.3.0 pin, cartopy, datashader,
    holoviews, scikit-learn); muphys depends on common[io] just for NetCDF.
13. **Deep namespace nesting**: `icon4py.model.atmosphere.subgrid_scale_physics.microphysics`.
14. **bindings asymmetry**: granule wrappers for diffusion/dycore/grid, functional muphys wrapper
    excluded from `all_bindings.py`; nothing for advection/microphysics. `tools/py2fgen` is
    fully generic (zero model deps).

### Assets to preserve

- Decomposition protocol design (singledispatch single-node/MPI split; GHEX isolated) — the
  best-factored subsystem.
- Advection's ABC + strategy + factory-function composition (the "modern" component template).
- Descriptive stencil naming (numbered-stencil migration completed); per-component `stencils/`.
- `setup_program` compile-time binding (`common/model_options.py`); math package.
- Serialbox datatests (numerical-equivalence oracle), StencilTest harness, nox/CI matrix
  (GH Actions CPU + CSCS GitLab GPU/MPI).
- ADRs 0001 (physics returns tendencies) & 0002 (declarative config) — accepted but
  unimplemented; this refactor implements them.

## Target Architecture

### Layers (tach-enforced)

```
apps        icon4py.driver, icon4py.validation, icon4py.fortran
components  icon4py.atmosphere.{dycore, diffusion, advection, microphysics.graupel,
            microphysics.muphys, diagnostics}     <- future: icon4py.ocean.*, icon4py.land.*
fields      icon4py.fields, icon4py.io, icon4py.testing
grid        icon4py.grid, icon4py.decomposition
core        icon4py.common                         (py2fgen: external dependency)
```

Layer semantics: higher layers import lower layers freely; same-layer edges must be declared
in tach.toml (so `dycore → diffusion` cannot merge accidentally); upward edges always fail CI.

### Distributions (6, down from 11; py2fgen leaves the repo)

| Distribution           | Import packages                                               | Replaces                                  |
| ---------------------- | ------------------------------------------------------------- | ----------------------------------------- |
| `icon4py-base`         | `icon4py.common`, `.grid`, `.decomposition`, `.fields`, `.io` | icon4py-common                            |
| `icon4py-atmosphere`   | `icon4py.atmosphere.*`                                        | 5 component dists                         |
| `icon4py-driver`       | `icon4py.driver`                                              | standalone_driver (legacy driver deleted) |
| `icon4py-testing`      | `icon4py.testing`                                             | half of icon4py-testing                   |
| `icon4py-validation`   | `icon4py.validation`                                          | other half of icon4py-testing (new)       |
| `icon4py-fortran`      | `icon4py.fortran`                                             | icon4py-bindings                          |
| *(external)* `py2fgen` | `py2fgen`                                                     | icon4py-tools — spun out to its own repo  |

### Module map

- **`icon4py.common`** (core): `dimension.py`, `constants.py`, `exceptions.py`,
  `field_type_aliases.py`, `precision.py` (ex-`type_alias`, frozen at import — see D8),
  `backends.py` (ex-`model_backends` + `model_options`), `math/` (unchanged), pruned `utils/`,
  `states/` (ModelState container + prognostic/diagnostic/tracer/tendency dataclasses +
  FieldMetaData — pure dataclasses over gt4py fields, no grid logic), `components.py` (the NEW
  minimal protocol below; current dead `Component` deleted).
- **`icon4py.grid`** (grid): current `grid/` + `topography/` + `external_parameters.py`;
  unified `domain.py` replacing horizontal/vertical Zone-Domain duplication (D3);
  `partitioning.py` ← `decomposition/halo.py` (halo construction is a grid concern; breaks the
  grid↔decomposition cycle).
- **`icon4py.decomposition`** (grid layer, below `icon4py.grid` via a declared one-way
  same-layer edge): `definitions.py` (DecompositionInfo, ProcessProperties, exchange/reduction
  protocols), `mpi_decomposition.py` (GHEX). singledispatch design preserved as-is.
- **`icon4py.fields`** (fields): interpolation + metrics + geometry-field *computation*
  (geometry data structures stay in grid), canonical CF-metadata name tables (`attrs`), and
  `registry.py` — the ~150-line replacement for `states/factory.py` (D1).
- **`icon4py.io`** (fields): ex-`common/io` + `monitor.py` (its only consumer); `[io]` extra
  slimmed to xarray+uxarray+netcdf4+cftime (D9).
- **`icon4py.atmosphere.dycore` / `.diffusion` / `.advection`** (components): current packages
  flattened, each keeping `stencils/`. No component imports another (tach-enforced).
  Advection's style is the template the others converge to.
- **`icon4py.atmosphere.microphysics.graupel`** (ex-microphysics) and **`.muphys`** (components):
  siblings behind the same Component protocol (D5). muphys standalone NetCDF runners →
  `icon4py.driver.apps`; `graupel_dace_hooks.py` stays internal to muphys.
- **`icon4py.atmosphere.diagnostics`** (components): ex-`common/diagnostic_calculations`
  (temperature/pressure diagnostics — atmosphere-specific; only the driver consumes it today).
- **`icon4py.driver`** (apps): ex-standalone_driver — `timeloop.py` (the single surviving copy
  of substepping logic), `config.py` (declarative TOML/JSON per ADR 0002), `provisioning.py`
  (generic component setup replacing `initialize_granules`), `initial_condition/`, `output.py`,
  `apps/` (each app = one composition: `apps/icon.py` full model, `apps/jw_test.py`,
  `apps/muphys_standalone.py`).
- **`icon4py.testing`** (fields layer): generic infra only — StencilTest, pytest hooks,
  fixtures, grid_utils, reference_funcs, parallel_helpers, locking, data_handling. Depends only
  on base. This makes the current false tach edge true.
- **`icon4py.validation`** (apps): ICON-reference oracle — `serialbox.py` savepoints,
  datatest_utils, the experiment registry (merging `testing/definitions.py` with the duplicated
  `standalone_driver/config.py` ExperimentConfig: validation maps experiment → driver config +
  reference data, depending on driver — resolving the circular-import hack), serialbox-based
  initializers (from the deleted legacy driver).
- **`icon4py.fortran`** (apps): ex-bindings — granule wrappers, `all_bindings.py` including
  muphys (fixes asymmetry), plus `namelist.py`: the ONLY ICON-namelist→config adapter in the
  codebase (D2).
- **`py2fgen`** (own repo): ex-`icon4py.tools.py2fgen`, top-level import `py2fgen`, zero
  icon4py deps; consumed by icon4py-fortran as an external dependency.

### tach.toml (end state sketch)

```toml
source_roots = ["base/src", "atmosphere/src", "driver/src",
                "testing/src", "validation/src", "fortran/src"]
exact = true
forbid_circular_dependencies = true
layers = ["apps", "components", "fields", "grid", "core"]

[[modules]]
path = "icon4py.common"
layer = "core"

[[modules]]
path = "icon4py.decomposition"
layer = "grid"

[[modules]]
path = "icon4py.grid"
layer = "grid"
depends_on = [{ path = "icon4py.decomposition" }]  # declared one-way

[[modules]]
path = "icon4py.fields"
layer = "fields"

[[modules]]
path = "icon4py.io"
layer = "fields"
depends_on = [{ path = "icon4py.fields" }]

[[modules]]
path = "icon4py.testing"
layer = "fields"
depends_on = [{ path = "icon4py.fields" }]

# components: separate modules, NO same-layer deps => isolation enforced
[[modules]]
path = "icon4py.atmosphere.dycore"
layer = "components"

[[modules]]
path = "icon4py.atmosphere.diffusion"
layer = "components"

[[modules]]
path = "icon4py.atmosphere.advection"
layer = "components"

[[modules]]
path = "icon4py.atmosphere.microphysics.graupel"
layer = "components"

[[modules]]
path = "icon4py.atmosphere.microphysics.muphys"
layer = "components"

[[modules]]
path = "icon4py.atmosphere.diagnostics"
layer = "components"

[[modules]]
path = "icon4py.driver"
layer = "apps"

[[modules]]
path = "icon4py.fortran"
layer = "apps"

[[modules]]
path = "icon4py.validation"
layer = "apps"
depends_on = [{ path = "icon4py.driver" }]

[[interfaces]]
expose = ["api"]
from = ["icon4py.atmosphere.*"]

[[interfaces]]
expose = ["api", "attrs"]
from = ["icon4py.fields"]

[external]
exclude = ["cupy", "ghex", "dace", "mpi4py", "serialbox"]  # true optionals only
rename = ["serialbox:serialbox4py"]
```

### Component interface (implements ADRs 0001/0002; replaces the dead protocol)

Three small pieces in `icon4py.common` — protocols + dataclasses, no framework:

**1. `ModelState`** (shared container; kills per-component InterpolationState/MetricState glue):

```python
@dataclasses.dataclass
class ModelState:
    prognostics: PrognosticStatePair  # double-buffered (now, next)
    diagnostics: DiagnosticState
    tracers: TracerState
    tendencies: TendencyState  # physics writes ONLY here (ADR 0001)
    prep_advection: PrepAdvection  # dycore->advection coupling; merges the two duplicates
```

**2. `Component` protocol + `StepInfo`** (one signature for every component):

```python
@dataclasses.dataclass(frozen=True)
class StepInfo:
    dt: wpfloat
    sim_time: datetime
    substep: int = 0
    n_substeps: int = 1
    first_timestep: bool = False
    # properties: at_first_substep, at_last_substep


class Component(Protocol):
    def __call__(self, state: ModelState, step: StepInfo) -> None: ...
```

Contract (documented + enforced by a shared contract test in `icon4py.testing`):
physics components write only `state.tendencies` + own diagnostics — prognostics bit-identical
before/after; dynamics components (dycore/diffusion/advection) may mutate prognostics in place;
substepping and `initial_run` special-casing become driver logic riding on `StepInfo`.
Rationale for `-> None` over returning tendency dicts: components write into preallocated
device fields (GPU-friendly, no per-step allocation); ADR 0001 is honored semantically.

**3. Static-field provisioning** (kills the 175-line manual mapping in `driver_utils`):

```python
@dataclasses.dataclass(frozen=True)
class DiffusionStaticFields(StaticFields):
    theta_ref_mc: fa.CellKField[wpfloat] = static_field("reference_potential_temperature_...")
    ...
```

`StaticFields.from_source(source)` resolves each declared canonical name (from
`icon4py.fields.attrs`) against a tiny `FieldGetter` protocol (`get(name)`). The driver becomes
a generic loop. Crucially, `FieldGetter` is satisfied by BOTH the current factory AND the new
registry — so component conversion (Phase 6) and factory replacement (Phase 7) are decoupled.
Granule boilerplate is reduced with helpers, not inheritance: `grid.domain_bounds(grid, dims)`
precomputes the start/end index table every `_determine_local_domains` rebuilds; a small
`bind_programs` helper wraps eager `setup_program` binding. Components keep explicit `__init__`s.

### Simplification decisions (D1–D9)

1. **Field factory → recipe registry** (`icon4py.fields.registry`, ~150 lines). One provider
   kind: a function registered with *explicit* dependency names + per-(grid, vertical, backend)
   memoization; `get(name)` resolves the DAG lazily. Whether a recipe wraps a gtx program,
   embedded field operator, or numpy is the recipe body's business — eliminates the 4-way
   provider taxonomy, runtime type-hint reflection, duplicated `_get_offset_providers`;
   `replace_khalfdim` collapses to one place (the registry's allocator). Keep the laziness,
   drop the framework.
2. **One config mechanism.** Component configs = plain frozen dataclasses with scientific
   defaults; no Fortran mirroring in core. Delete `config/options.py`, `utils/fortran_config.py`,
   every `from_fortran_dict`. The single ICON-namelist adapter lives in `icon4py.fortran.namelist`
   (the only place receiving namelist values at runtime); validation reuses it.
3. **Unify Domain/Zone** in `icon4py.grid.domain`: one generic `Domain` + one `domain(dim)(zone)`
   API; horizontal and vertical zone vocabularies stay separate enums behind a common protocol
   (they are semantically different; the machinery isn't). Move `num_levels` out of horizontal
   `GridConfig`; `Grid` and `VerticalGrid` become peers composed by the driver.
4. **Kill the legacy driver**; its one unique capability (serialbox initialization) moves to
   `icon4py.validation.initializers` and plugs into the surviving driver.
5. **Microphysics: siblings behind one protocol, no forced merge** — they are different
   scientific schemes; merging internals is meaningless. muphys gets a ~50-line Component
   wrapper; graupel converts to tendency-writing. A parametrized validation test runs both
   through the identical harness — the real interchangeability proof.
6. **Split testing**: `icon4py.testing` (infra, fields layer) vs `icon4py.validation`
   (ICON-reference oracle, apps layer). Fixes the false tach edge, removes driver→testing.
7. **py2fgen spin-out**: rename to top-level `py2fgen` dist (zero deps, tach-verified), then
   extract to its own repository via `git filter-repo`; icon4py-fortran consumes it as an
   external dependency (git/PyPI).
8. **Precision & singletons without global mutation**: delete `set_precision()`;
   `icon4py.common.precision` resolves wpfloat/vpfloat once at import from `ICON4PY_PRECISION`
   (bindings, as process entry point, set it pre-import; the pytest plugin maps
   `--enable-mixed-precision` onto it before collecting). Module-level exchange/reduction
   singletons → `decomposition.single_node()` factory; `exchange` is an explicit ctor arg.
9. **IO slimming**: drop cartopy/datashader/holoviews/scikit-learn from the extra (viz →
   docs notebooks or deleted); attempt lifting the uxarray pin during the move.

## Refactoring Plan (phases; each lands with green CI)

Phases 0–5 must be **bit-identical** to baseline (pure moves/renames/mechanical extraction);
Phases 6–7 are behavior-adjacent and get dedicated parity harnesses. Sequencing: 0→1→2→3→4
sequential; after 4, Phases 5 and 8 parallelize freely; 6 and 7 are mutually independent (via
`FieldGetter`); 9 last. Critical path: 0–1–3–4–6.

**Phase 0 — Honest baseline + guardrails** *(XS, no risk)*

- Fix tach.toml to reflect reality: add the true `testing → {advection, diffusion, dycore, microphysics, standalone_driver}` edges (documented as debt), standalone_driver's real deps,
  and the legacy driver's `testing` edge.
- Fix the packaging bug: declare advection + microphysics in standalone_driver's pyproject.
- Add `tach check` (module graph, not just check-external) to pre-commit + CPU CI.
  **Known issue found while landing this**: tach >=0.27 (up to at least 0.35) cannot resolve
  first-party imports across multiple source roots sharing the `icon4py` namespace package,
  so `tach check` silently sees no internal imports (`check-external` is unaffected). The
  `tach check` hook is therefore pinned to tach 0.26.1 until this is fixed upstream; the
  regression should be reported to gauge-sh/tach, since the end-state `layers`/`interfaces`
  enforcement depends on `tach check`.
- Add a **composition smoke test**: 2-timestep dycore-only run (JW test, small grid) as a fast
  CI job — the continuous proof of pluggability for every later phase.

**Phase 1 — Extract `icon4py-validation`** *(M–L, low risk: pure code motion)*

- New dist `validation/`: move `serialbox.py`, `datatest_utils.py`, experiment registry from
  `testing/definitions.py`, data-download URL registry. `icon4py.testing` keeps
  StencilTest/fixtures/hooks/grid_utils/reference_funcs.
- Merge the duplicated ExperimentConfig (testing/definitions.py ↔ standalone_driver/config.py):
  driver owns run-config dataclasses; validation's registry maps experiment → driver config +
  reference data. Remove `icon4py-testing` from driver deps (the TODO).
- Mechanical import updates across all component `tests/` dirs.
- Verify: full nox matrix + datatests unchanged.

**Phase 2 — Kill the legacy driver** *(S–M, low risk)*

- Port serialbox initialization → `icon4py.validation.initializers`, pluggable into the
  surviving driver; migrate legacy-only datatests; delete `model/driver/`.

**Phase 3 — Split the god package; introduce tach layers** *(XL but mechanical; risk = merge churn)*

- `icon4py.model.common` → `icon4py.{common, grid, decomposition, fields, io}` inside the
  renamed `icon4py-base` dist. One PR per target package, in dependency order (common →
  decomposition → grid → fields → io); each PR = `git mv` + repo-wide import rewrite +
  tach.toml update in the same PR.
- Break the grid↔decomposition cycle: `decomposition/halo.py` → `icon4py.grid.partitioning`;
  push the `grid.base` primitive that `decomposition/definitions.py` needs into `icon4py.common`.
- `diagnostic_calculations`: stays in base during this phase; moves to
  `icon4py.atmosphere.diagnostics` in Phase 4 when the atmosphere dist exists.
- Delete dead `components/components.py` (keep `monitor.py` → moves with io) and
  `config/options.py`.
- Introduce `layers = [...]` + per-module `layer =` for base packages.
- Coordination: short merge freeze per sub-PR; no compat shims (monorepo updates atomically;
  pre-1.0 external users get release notes).
- Verify: bit-identical datatests; CSCS pipeline paths updated in-PR.

**Phase 4 — Flatten components, consolidate dists, rename driver, spin out py2fgen** *(L, mechanical)*

- `atmosphere/*` → single `icon4py-atmosphere` dist: `icon4py.atmosphere.{dycore, diffusion, advection, microphysics.graupel, microphysics.muphys, diagnostics}`.
- `standalone_driver` → `icon4py-driver` / `icon4py.driver`; muphys NetCDF runners →
  `icon4py.driver.apps`.
- `bindings` → `icon4py-fortran` / `icon4py.fortran`.
- py2fgen: rename to top-level `py2fgen` dist/import, then **extract to its own repository**
  (`git filter-repo` preserving history); icon4py-fortran consumes it as external dep.
- tach: full target module list (interfaces deferred to Phase 9); component same-layer
  isolation enforced from here on. nox sessions / CI matrices / GitLab pipeline renamed.
- Verify: bit-identical datatests; `uv lock` proves the dist graph is complete (no undeclared
  workspace imports).

**Phase 5 — Grid domain unification + granule helpers** *(M, low-medium risk)*

- Unified `icon4py.grid.domain`; delete `DomainType` TypeVar; `num_levels` out of `GridConfig`.
- Add `grid.domain_bounds()` + `bind_programs`; convert each component's boilerplate (one PR
  per component, parallelizable).
- Verify: bit-identical datatests (index computation must not change results).

**Phase 6 — Component protocol + shared state + auto-provisioning** *(XL, highest scientific risk — one PR per component)*

- Land `ModelState`, `StepInfo`, `Component`, `StaticFields`/`static_field`, `FieldGetter`
  (satisfied by the existing factory, so this phase does not wait for Phase 7).
- Convert in order of confidence: **diffusion** (simplest) → **dycore** (substep flags via
  StepInfo; PrepAdvection moves to ModelState) → **advection** → **graupel** (in-place →
  tendency-writing: internal scheme computation unchanged, only the outer update application
  moves) → **muphys** (new Component wrapper class).
- Replace `driver_utils.initialize_granules` with the generic provisioning loop; delete
  per-component InterpolationState/MetricState as each component converts.
- Add the physics contract test (prognostics untouched) parametrized over both microphysics.
- Numerics: dynamics conversions are signature-only → bit-identical required. Graupel's
  tendency conversion may change bit patterns (`x + dt*t` vs direct update) → validate within
  existing datatest tolerances; any tolerance introduction documented with max observed
  deviation. Old entry points survive one phase as thin delegating aliases (no flag day).

**Phase 7 — Recipe registry replaces the field factory** *(L, medium technical risk)*

- Implement `icon4py.fields.registry`; port recipes package-by-package (geometry →
  interpolation → metrics), coexisting with the old factory.
- Parity harness: every canonical field computed via old factory AND new registry, assert
  allclose at tight tolerance; existing per-field serialbox datatests re-pointed at the registry.
- Flip `fields.combine()` to the registry; delete `states/factory.py` + provider machinery +
  `replace_khalfdim` shims.

**Phase 8 — Config, precision, singletons, IO slimming** *(M, low risk; parallel with 6–7)*

- Delete `utils/fortran_config.py` + all `from_fortran_dict`; add `icon4py.fortran.namelist`;
  validation switches to it. Driver config → declarative TOML/JSON per ADR 0002.
- `precision.py` freeze; delete `set_precision`; bindings/pytest set env pre-import.
- `decomposition.single_node()` factory; remove module-level singletons and default-arg usage.
- IO extra slimming + uxarray unpin attempt.

**Phase 9 — Lock it in** *(S)*

- tach `[[interfaces]]` (components + fields expose `api`); shrink `[external] exclude` to true
  optionals (undeclared-dep bug class now fails CI).
- Add muphys to `all_bindings.py` FUNCTIONS; advection/graupel wrappers only if ICON-side
  demand exists (explicitly out of scope otherwise).
- New ADR for layers + component protocol; amend ADR 0001 ("writes into preallocated
  tendencies"); rewrite READMEs / CLAUDE.md / CODING_GUIDELINES test-layout references.

## Verification

**Every phase:** `tach check` + `tach check-external` in pre-commit and CI — tach.toml updated
in the same PR as every move so the architecture file never drifts; full nox CPU matrix
(embedded/dace_cpu/gtfn_cpu × py3.10/3.14 × components); CSCS GPU+MPI pipeline on each phase's
closing PR; StencilTests throughout (move-insensitive, catch stencil breakage); the Phase-0
composition smoke test on every PR.

**Numerical safety net:** serialbox datatests are the regression oracle. Phases 0–5: results
bit-identical. Phase 6: per-component equivalence via existing datatests (+ documented
tolerances only where the graupel tendency rewrite requires them). Phase 7: field-level parity
harness old-factory-vs-registry, retired when the factory is deleted.

**New enforcement added:** tach layers (upward import = CI failure), component same-layer
isolation, `[[interfaces]]` public-API surface, external-dep correctness, physics contract test
(prognostics immutability), mypy coverage growing with every new module (new code typed from
day one).

## Critical files

- `tach.toml` — evolves every phase; end state above.
- `model/common/src/icon4py/model/common/states/factory.py` — replaced by `icon4py.fields.registry` (Phase 7).
- `model/common/src/icon4py/model/common/grid/{horizontal,vertical}.py` — Domain/Zone unification (Phase 5).
- `model/common/src/icon4py/model/common/components/components.py` — deleted; new protocol in `icon4py.common.components` (Phases 3/6).
- `model/testing/src/icon4py/model/testing/{serialbox,datatest_utils,definitions}.py` — validation split (Phase 1).
- `model/standalone_driver/src/icon4py/model/standalone_driver/{driver_utils,config}.py` — provisioning + config merge (Phases 1/6).
- Root `pyproject.toml` + per-package pyprojects — dist consolidation (Phase 4).
- `bindings/src/icon4py/bindings/all_bindings.py` — muphys inclusion (Phase 9).
