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

- [uv](https://docs.astral.sh/uv/) — the entry point uses `uv run --isolated --group scripts`
  so all scripts dependencies declared in the scripts dependency group (e.g. `typer`) are installed
  automatically into an ephemeral isolated environment.
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
│   ├── _<...>.sh            # Utility: helper functions, not a script
│   ├── <...>.sh             # Script: setup-env sub-command
│
└── tests/
    ├── python/              # Python scripts tests
    │   ├── conftest.py      # Shared pytest fixtures
    │   └── test_<...>.py    # Command test
    └── sh/
        └── test_lib.bats    # Bash library tests (bats-core)
```

## Conventions

| Convention    | Meaning                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `_` prefix    | Shared infrastructure, **not** a sub-command (`_lib.sh`, `_cli.py`, `_util.py`) |
| `python/*.py` | Each non-prefixed python module defining a `cli` typer app is auto-discovered   |
| `sh/*.sh`     | Each non-prefixed script is auto-discovered and wrapped as a Typer sub-command  |

## Adding a New Sub-Command

### Python

1. Create `python/my_tool.py` defining a `cli` Typer app.
2. If needed, import `_common` or other shared helper module.
3. If possible, add tests in `tests/python/test_my_tool.py`.
4. (Optional) If you also want the script to be directly runnable without going through
   the common entry point, you need to add the following changes:
   - use the same *she-bang* as the `run` script
   ```bash
   #!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
   ```
   - use absolute imports and add `/scripts/` to the `sys.path` (when run as a isolated script)
   ```python
   if __name__ == "__main__":
       sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
   from python import (
       _common as common_common,
   )  # Relative imports like `from . import _common` will fail
   ```

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
