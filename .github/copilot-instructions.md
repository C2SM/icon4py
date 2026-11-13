<!--
Restates the review-relevant subset of CODING_GUIDELINES.md and AGENTS.md on
purpose: Copilot code review reads only .github/ files (from the PR base
branch, ~4k chars each) and never those two. PR Agent reads this file first
(see .pr_agent.toml). Keep short.
-->

# Code review rules

## Always flag

- Generic code added to a component instead of `model/common`. If what the
  code does can be described with only grid entities (cell, edge, vertex,
  level) and mathematical operations (gradient, divergence, interpolation,
  average, tendency), it belongs in `model/common`.
- Generic code named after its caller's variables (`vn`, `theta_v`, a
  tracer) rather than after the operation, or a near-duplicate of an
  existing `model/common` stencil or helper. Stencil names follow
  `docs/stencil_naming_convention.md`.
- New or changed code paths without a test in the component's
  `tests/<component>/` tree (`unit_tests/`, `stencil_tests/`,
  `integration_tests/`, `mpi_tests/`). `stencil_tests/` is only for GT4Py
  stencils that may be integrated into ICON.
- Off-by-one errors in domain bounds (`h_grid.Zone`, vertical level ranges,
  start/end indices) and in connectivity offsets.
- A field written by a stencil and then read through a neighbor
  connectivity without a halo exchange in between, and reductions that mix
  owned and halo entries.
- A hard-coded `float64`/`float32` where the surrounding code uses
  `wpfloat`/`vpfloat` from `icon4py.model.common.type_alias`, which breaks
  the single-precision mode.
- Error messages that are not a single sentence starting with a capital
  letter and ending with a period, with code objects and interpolated
  strings in single quotes.
- Secrets, tokens, hostnames, or per-developer absolute paths committed in
  source.

## Don't comment on

- Formatting or import order — `ruff format` / `ruff check` own that, and
  pre-commit + CI enforce it.
- Missing license headers — pre-commit inserts them automatically.
- Missing docstrings on private or internal helpers.
- Anything `tach`, `ruff`, or `mypy` already enforces deterministically; CI
  will catch it without a comment.

## Review style

- One comment per distinct issue; lead with the risk, then the fix.
- Cite evidence as `path/to/file.py:LINE`.
- When a PR changes architecture or adds an architecture-level feature,
  question the design and propose refactorings where warranted. For a
  localized fix, prefer the smallest change over a refactor.
- If a finding is a question rather than a defect, phrase it as one.
