# ICON4Py Agent Instructions

ICON4Py is a Python implementation of the Fortran [ICON climate and weather model](https://www.icon-model.org/). The upstream open source release is available at https://gitlab.dkrz.de/icon/icon-model.

## Monorepo structure

uv workspace with 10 namespace packages. All share the `icon4py` namespace. Source lives under `<package>/src/icon4py/...`. Packages are installed editable by `uv sync`.

```
model/
  atmosphere/
    advection/          # icon4py.model.atmosphere.advection
    diffusion/          # icon4py.model.atmosphere.diffusion
    dycore/             # icon4py.model.atmosphere.dycore
    subgrid_scale_physics/
      microphysics/     # icon4py.model.atmosphere.subgrid_scale_physics.microphysics
      muphys/           # icon4py.model.atmosphere.subgrid_scale_physics.muphys
  common/               # icon4py.model.common  ← shared code, all model packages depend on this
  driver/               # icon4py.model.driver  ← depends on diffusion, dycore, common, testing
  standalone_driver/    # icon4py.model.standalone_driver
  testing/              # icon4py.model.testing ← pytest plugin, fixtures, serialbox helpers
tools/                  # icon4py.tools ← Fortran integration (py2fgen CLI), independent of model
```

Tach enforces the dependency graph in `tach.toml`. All model atmosphere packages and standalone_driver depend only on `common`. Driver depends on diffusion + dycore + common + testing. Tools is independent.

**Always run `uv sync` from the repo root.** Running it from a subpackage only installs that package's deps.

## Environment setup

A `shell.nix` provides system dependencies (MPI, Boost >= 1.85, GCC, cmake, etc.). Python is managed by uv (see `.python-version`), not nix. The venv must be created inside the nix shell so that native packages (mpi4py, ghex) can find the system libraries.

`shell.nix` sets `UV_MANAGED_PYTHON=1` to prevent uv from falling back to system Python interpreters. uv reads `.python-version` to determine the Python version. For a clean rebuild from scratch:

```bash
nix-shell shell.nix --run "rm -rf .venv && uv sync --extra all --no-cache"
```

At the start of every session:

```bash
nix-shell shell.nix --run "uv sync --extra all"
```

Nix shell sets key environment variables:

- `UV_MANAGED_PYTHON`: forces uv to use managed Python only
- `ICON4PY_TEST_DATA_PATH`: where serialized test data is stored
- `PYTEST_ADDOPTS`: default pytest options (xdist workers, verbosity)
- `LD_LIBRARY_PATH`: MPI library path
- `SSL_CERT_FILE`: certificates for data download
- `GT4PY_BUILD_CACHE_DIR`: GT4Py stencil compilation cache location
- `GT4PY_BUILD_JOBS`: not set by default; set it to limit parallel stencil compilation jobs

uv caches built wheels by package hash, not just Python version. If native extensions fail to load (e.g. `ModuleNotFoundError` for a `.so`), the cache may contain a build for a different Python version. Fix: re-run the clean rebuild command above.

## Pre-commit checks

All checks (ruff lint, ruff format, tach, mypy, license headers, toml/yaml formatting) run through pre-commit. Must run inside nix-shell:

```bash
uv run --group dev --frozen --isolated pre-commit run --all-files
```

Notes:

- License headers are auto-added by the `insert-license` hook using `HEADER.txt`. Do not add them manually.
- mypy uses `--group typing-distributed` (includes mpi4py stubs), not just `--group typing`.
- mypy only runs on specific paths listed in `pyproject.toml [tool.mypy].files`. Running mypy on the whole codebase will produce errors from unchecked packages.
- ruff excludes `docs/` and `examples/` directories.

## Tests

All test commands must run inside `nix-shell shell.nix`.

### Test categories

| Category      | Marker / flag               | Needs serialized data?    | Typical runtime        |
| ------------- | --------------------------- | ------------------------- | ---------------------- |
| Unit tests    | none (no `datatest` marker) | No                        | Fast (\<1s per test)   |
| Data tests    | `@pytest.mark.datatest`     | Yes (auto-downloaded)     | Slow (2+ min per file) |
| Stencil tests | `-k stencil_tests`          | No (uses generated grids) | Moderate               |
| MPI tests     | `@pytest.mark.mpi`          | Yes (auto-downloaded)     | Slow                   |

### Running tests

```bash
# Fast unit tests (no data required):
uv run --group test --frozen pytest model/<component>/tests/<component>/unit_tests/

# Skip datatests:
uv run --group test --frozen pytest --datatest-skip model/<component>/

# Only datatests (requires test data):
uv run --group test --frozen pytest --datatest-only model/<component>/

# MPI tests (requires mpi4py, distributed extra; always use -n0 for sequential):
mpirun -np 4 ci/scripts/ci-mpi-wrapper.sh uv run --group test --frozen pytest -v -s --with-mpi -n0 -k mpi_tests model/<component>/
#   --with-mpi: enables MPI test mode (from pytest-mpi plugin)
#   -k mpi_tests: selects tests by directory/class name convention
#   ci-mpi-wrapper.sh: suppresses stdout from non-zero ranks

# Sequential mode (reduce memory pressure):
uv run --group test --frozen pytest -n0 <paths>
```

### Custom pytest options

Registered by `icon4py.model.testing.pytest_hooks` (auto-loaded via `addopts`):

| Option                            | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| `--datatest-only`                 | Run only `@pytest.mark.datatest` tests                                   |
| `--datatest-skip`                 | Skip all datatests                                                       |
| `--backend <name>`                | GT4Py backend (default: roundtrip; others: gtfn_cpu, gtfn_gpu, embedded) |
| `--grid <name>`                   | Grid to use                                                              |
| `--enable-mixed-precision`        | Switch from double to mixed-precision                                    |
| `--level {any,unit,integration}`  | Filter by `@pytest.mark.level` marker                                    |
| `--skip-stenciltest-verification` | Skip verification of StencilTest against reference outputs               |

### Test directory convention

Each component's tests live under `tests/<component_name>/` with subfolders:

```
tests/<component>/
  unit_tests/         # fast, no external data
  stencil_tests/      # GT4Py stencil integration tests (for icon-exclaim)
  integration_tests/  # end-to-end tests
  mpi_tests/          # parallel/MPI tests
  conftest.py
  fixtures.py
```

### Fast test paths

No serialized data required, complete in seconds:

- `model/common/tests/common/decomposition/unit_tests/` — ~74 tests, ~6s

### Benchmarking

`pytest-benchmark` is active by default. Use `--benchmark-disable` to skip benchmarks during regular test runs.

### Fixture imports

Fixtures re-exported through a component's `fixtures.py` must be explicitly imported in the test file, even if not used directly in the test body. Pytest resolves fixtures from the test module's namespace, not transitively through other fixtures. Missing imports cause `fixture 'X' not found` errors.

## GT4Py stencil compilation

GT4Py compiles stencils to C++ on first use. This is slow. Compiled code is cached in `GT4PY_BUILD_CACHE_DIR` (set by shell.nix to `~/.cache/gt4py`). To limit parallel compilation jobs:

```bash
export GT4PY_BUILD_JOBS=4
```

## Nox (CI-style test sessions)

Nox mirrors the CI pipeline. Useful for running comprehensive test suites:

```bash
# List sessions:
uv run --group test --frozen nox -l

# Run all tests for a specific component and subset:
uv run --group test --frozen nox -s 'test_common(datatest=True)'
uv run --group test --frozen nox -s 'test_common(datatest=False)'
```

Subset options: `datatest`, `stencils`, `basic` (datatest-skip, no stencils/benchmarks).
