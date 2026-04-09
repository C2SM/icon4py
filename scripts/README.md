# scripts/ — Dev-Scripts Toolbox

A unified CLI for project development scripts, supporting Python, Bash, and
other recipes through a single entry point.

## Quick Start

```bash
# Show all available commands
./scripts/run --help

# Run any command
./scripts/run my-command --option arg
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — the entry point uses `uv run --script`
  with PEP 723 inline metadata, so dependencies (e.g. `typer`) are installed
  automatically into an ephemeral environment.
- [bats-core](https://github.com/bats-core/bats-core) — only needed to run
  Bash tests.

## Layout

```
scripts/
├── run                      # Single entry point (uv shebang)
│
├── python/                  # Python scripts (importable package)
│   ├── __init__.py
│   ├── _common.py           # Shared helpers (logging, paths, subprocess)
│   ├── _<...>.py            # Utility: helper module, not a script
│   └── <...>.py             # Script: contains a `cli = typer.Typer()` global symbol
│
├── sh/                      # Bash scripts
│   ├── _lib.sh              # Shared functions (sourced, not executed)
│   ├── setup_env.sh         # Example: setup-env sub-command
│   └── cleanup.sh           # Example: cleanup sub-command
│
└── tests/
    ├── python/              # Python scripts tests
    │   ├── conftest.py      # Shared pytest fixtures
    │   └── test_<...>.py    # CLI structure tests
    └── sh/
        └── test_lib.bats    # Bash library tests (bats-core)
```

## Conventions

| Convention    | Meaning                                                                         |
|---------------|---------------------------------------------------------------------------------|
| `_` prefix    | Shared infrastructure, **not** a sub-command (`_lib.sh`, `_cli.py`, `_util.py`) |
| `python/*.py` | Each non-prefixed python module defining a `cli` typer app is auto-discovered   |
| `sh/*.sh`     | Each non-prefixed script is auto-discovered and wrapped as a Typer sub-command  |

## Adding a New Sub-Command

### Python

1. Create `python/my_tool.py` defining a `cli` Typer app.
2. Import `_common` at the top for shared helpers if needed.
3. Add tests in `tests/test_my_tool.py`.

### Bash

1. Create `sh/my_tool.sh` (it will be auto-discovered).
2. Source `_lib.sh` at the top for shared helpers if needed.
3. Add tests in `tests/sh/test_my_tool.bats`.

## Running Tests

```bash
# Python tests
cd scripts && uv run --group test pytest

# Bash tests
bats scripts/tests/sh/
```
