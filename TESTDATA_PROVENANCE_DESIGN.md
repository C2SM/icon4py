# Design: provenance and drift detection for icon4py serialized test data

- **Status**: draft, local working document, not for commit
- **Author**: Jacopo Canton (@jcanton), drafted with Claude
- **Created**: 2026-07-30

**(Wh)Y-statement**: In the context of regenerating Serialbox test data from an
independently-evolving ICON Fortran upstream, facing the problem that a regeneration
silently absorbs unrelated upstream changes and surfaces them days later as
unexplained CI failures, we decided to record machine-readable metadata in every
archive and to mechanically diff each new archive against its predecessor, and
neglected pinning the upstream ICON commit, to achieve fast attribution of every data
delta, accepting that the mechanism reports rather than blocks.

## Context

### The incident that motivates this

icon4py datatests validate against Serialbox dumps produced by an instrumented ICON
Fortran build, packaged per `(experiment, comm_size)` and published to
`https://rgw.cscs.ch/c2sm:testdata`.

Data was regenerated v05 → v06 to add savepoints for one feature. Between v05
(2026-06-09) and v06 (2026-06-24), unrelated upstream ICON commits had landed. One of
them — `a9435ae531`, 2026-06-18, *"[nwp] Improve wave energy propagation near the
coast"* — moved the `primal_normal_cell` / `dual_normal_cell` computation in
`complete_patchinfo` from an `rl_start=2` loop into the `rl_start=1` loop (with
per-neighbour `IF (ilc > 0 .AND. ibc > 0)` guards), and changed
`calculate_tangent_plane_at_edge` from `i_rcstartlev = 2` to `rl_start = 1` (with an
`IF (ilc==0 .OR. ibc==0) CYCLE` guard). Both in `src/shr_horizontal/mo_intp_coeffs.f90`.

Consequence: three reference fields (`primal_normal_cell`, `dual_normal_cell`,
`pos_on_tplane_e`) that were zero on the outermost lateral-boundary edge row of the
limited-area MCH grid now carry real values there. Three datatests started failing.
Attribution took a multi-hour manual git archaeology session.

### Generalised failure mode

> We regenerate serialized data expecting delta X. Upstream has moved on, so we
> silently get X + Y + Z. Y and Z surface later as mystery failures, and no recorded
> provenance connects them back to an upstream commit.

### Concerns

- Regeneration happens a few times a year, by one or two people, under time pressure,
  on a slurm cluster, often at the end of a long feature branch. Ceremony will be
  skipped or will bit-rot.
- A generation campaign is 18 slurm tasks over multiple hours. A failure in one task
  must not discard the others.
- The generating machine (CSCS Santis) is not where icon4py development happens.
- Data volume is ~250 GB. Re-publishing history is not viable.

## Facts established during investigation

Verified against the working tree and against local archives, not assumed.

1. **ICON already emits full VCS provenance, and it is already inside every archive.**
   `mo_util_vcs.f90:169` `show_version` is called unconditionally at startup and writes
   to the slurm log, which `copy_ser_data()` already archives:

   ```
    executable: /capstor/store/cscs/userlab/cwd01/jcanton/icon-exclaim.serialize/build_serialize/bin/icon
    date: 20260706
    time: 135858
    user: Jacopo Canton, ETHZ (jcanton)
    host: nid005281 (Linux 6.4.0-150600.23.53-64kb aarch64)
    version: 2026.04
    revision: icon-2025.10-dwd-2.0-331-gbebe90513b1da6842d0609579343af57b63a8158
    repository: git@gitlab.dkrz.de:icon/icon-nwp.git
    local branch: serialize_tmx_sfc
    model components:
      ICON-Land:
        revision: icon-land-2026.04-18-geb24d7431a16ef66bae693a7ef95c32ca47862ff
   ```

   ICON's own fields carry exactly one leading space; nested externals are indented
   further. That is the parser anchor.

2. **Serialbox already digests every field it writes.**
   `ser_data/ArchiveMetaData-icon_pydycore_rank{N}.json` carries
   `fields_table[field] = [[byte_offset, digest], ...]`, and
   `ser_data/MetaData-icon_pydycore_rank{N}.json` maps savepoints to field occurrences.
   An exact per-field fingerprint of an 11 GB archive costs two JSON reads.

   Three measured caveats, all of which break a naive implementation:
   - **`fields_table[field][i][0]` is a byte offset, not an identifier.** The occurrence
     index from `fields_per_savepoint` is the *list position*. Keying a dict on element 0
     and indexing it by the occurrence raises `KeyError` on the first repeatedly written
     field.
   - **The digests must be used verbatim.** They are 32-byte tokens printed byte by byte
     with leading zeros stripped *per byte*, so they are 54-64 characters long. Measured
     over 46 422 digests in the local archives: 12.5 % are 64 characters and 0.34 % start
     with `0` — consistent with per-byte stripping (1/256) and refuting both zero-padded
     hex (1/16) and a single global strip (which would leave 93.75 % at 64). `zfill(64)`
     only realigns the first byte and corrupts the rest. They are also *not*
     `hashlib.sha256` values despite the `hash_algorithm` field; treat them as opaque.
   - **Fingerprints must be keyed per rank.** Savepoint structure, ordering, metainfo and
     field sets are identical across ranks, but the values are not (only ~7 % of records
     match between ranks — the scalars). Ranks cannot be merged into one namespace.

   Other verified invariants across all 27 local rank-archives: occurrence indices are
   contiguous `0..N-1`; `field_map` is a *superset* of `fields_table` (two declared but
   never-written fields exist), so iterate `fields_per_savepoint`, never `field_map`; the
   same `(field, occurrence)` is legitimately shared by several savepoints, so field name
   alone is never a key; and `date` metainfo changes on every regeneration, so keys must
   use the occurrence ordinal, not the date.

3. **`version` is already a per-experiment field.** `ExperimentDescription.version: int = 6`
   (`definitions.py:159`) is a *default* that all six experiments inherit by omission.
   Per-experiment versioning is a matter of spelling out the values, not of redesign.

4. **Namelist state, including defaults, is already fully archived.**
   `NAMELIST_ICON_output_atm` and its `.json` are ICON's *post-read* dump: 37 sections,
   every variable with its effective value, defaults included (`grid_nml` alone carries
   25 keys). `NAMELIST_<exp>_sb` / `NAMELIST_expname.json` carry the user-set input and
   already contain the resolved pool paths (`dynamics_grid_filename = "/capstor/.../
   icon_grid_0013_R02B04_R.nc"`), as does the output dump (`EXTPAR_FILENAME` too).

   Therefore an upstream change to a **namelist default**, and a change to the **grid or
   extpar file used**, are both already detectable from files present in every archive
   generated to date. No archive change is required — only a consumer that diffs them.
   *(This supersedes an earlier draft that proposed archiving `nml.atmo.log` and
   `run/exp.<name>_sb`. Both are redundant.)*

5. **Downloads are unverified.** `pooch.retrieve(..., known_hash=None)`
   (`data_handling.py:66`). A `.extraction_complete` marker short-circuits re-extraction
   before the lock is taken, so an already-extracted checkout never observes a change to
   the published archive.

6. **The HackMD HOWTO is partly stale.** It links the rgw archive at the top, but the
   body still describes Polybox upload and `version = 3`.

## Decision

Adopt a **report-first** design with three parts, in this order of importance:

1. **Reduce blast radius** — per-experiment versioning, so a regeneration for one
   experiment cannot invalidate the other five.
2. **Record provenance** — one `archive_metadata.json` per archive, entirely
   machine-harvested, with a `provenance` section and room to grow.
3. **Diff the data** — a `serdata` tool that fingerprints an archive from Serialbox's
   own hashes and reports what changed against the previous version, alongside the
   namelist diff and the scoped upstream commit list.

### Terminology: savepoint classes

Serialbox savepoints fall into classes that must be judged by different standards. The
tool classifies each savepoint into exactly one:

- **`initial-state`** — savepoints with `location=initial-state`, i.e. `prognostics` and
  `diagnostics`. Kept separate so a legitimate testcase or IC change does not read as a
  static change.
- **`static`** — written once at startup, describing time-invariant model state:
  `icon-grid`, `metric-state`, `interpolation-state`, `smooth-topo-savepoint`,
  `tmx-init`. A change here is almost always a real semantic change in ICON. This is
  precisely where the wave commit landed. **Every changed field is named and must be
  justified.**
- **`evolving`** — everything else. Reported as counts and a first divergence point only;
  never gated, because a static change legitimately propagates into all of it and
  toolchain noise lives here too.

**`initial-state` is tested before `static`.** The initial-state savepoints carry neither
a `date` nor a `dyn_timestep`, so a "written once" rule alone would swallow them and leave
the initial-state class permanently empty. Conversely `jabw-initial-state-exit` is named
like an initial state but carries a `date`, and is correctly `evolving`.

Membership is an explicit list in the source. The auto-rule (*no `date` and no
`dyn_timestep` metainfo*) is used only to **flag new savepoints for classification**,
never to classify silently. A per-experiment floor on the `static` record count is
asserted, so the report cannot go green because a savepoint vanished or gained a `date`.

Measured record counts, summed over all ranks, from the local archives (these become the
asserted floors):

| archive | static | initial-state | evolving |
|---|---|---|---|
| mpitask1_exclaim_ape_R02B04_v05 | 166 | 25 | 1842 |
| mpitask1_exclaim_ch_r04b09_dsl_v05 | 173 | 25 | 1881 |
| mpitask1_exclaim_gauss3d_v05 | 170 | 23 | 2201 |
| mpitask1_exclaim_nh35_tri_jws_v05 | 166 | 23 | 6611 |
| mpitask1_exclaim_nh_weisman_klemp_v05 | 166 | 25 | 9116 |
| mpitask4_exclaim_ape_aesPhys_v05 | 664 | 100 | 26532 |
| mpitask4_exclaim_ape_aesPhys_v06 | 752 | 100 | 27432 |

**Positive control.** Running the classifier over the real `exclaim_ape_aesPhys` v05→v06
regeneration, which added the TMX turbulence scheme: `static` 0 of 664 changed,
`initial-state` 0 of 100 changed, and every one of the 15 555 changed records is
`evolving`, alongside 9 added savepoints and 85 added fields. The classifier separates
"must not change" from "expected to change" cleanly on a real regeneration.

### Design rules

- **A failure is isolated to the task that caused it.** One `(experiment, comm_size)`
  task may fail loudly and stop — the other tasks, and the remaining comm sizes, carry
  on. Today `run_experiment` raises and `future.result()` re-raises out of the
  `for comm_size` loop, discarding the rest of a multi-hour campaign.
- **Baselines are recomputed from `ser_data/`, never read from a file a previous run
  had to write.** To report a diff you need the *old* archive's fingerprint. It is
  computed on the spot from the old archive's own `ArchiveMetaData-*.json`, which every
  archive ever produced already contains. The rejected alternative — have each run write
  a fingerprint file for the next run to read — fails on the first run under the new
  scheme, which is exactly the run where drift bites: no predecessor ever wrote the file,
  so the check reports "no baseline, skipped" and exits green.
- **Every new parser ships with a fixture test** runnable offline on a laptop.
- **Upload stays manual**, and the local verification run happens before it.

### Decision: upload ownership

The script does not upload. The workflow is:

```
generate → serdata diff report → run datatests against the fresh data → human uploads → bless
```

Rationale: the value of an unpublished archive is that it can be rejected. Verifying
before publication is the point; automating the upload would put publication *before*
the strongest gate. The cost is that "refuse to publish" is a human decision rather than
a structural one — covered by the lockfile CI check (Tier 1.3), which blocks the PR that
would make a bad archive canonical.

### Out of scope

- **CI comm-size coverage.** `mpitask2` archives are generated and published but never
  resolved, because `.test_runner_mpi` hardcodes `ICON4PY_TEST_MPI_SUBCOMM_SIZE: 4`.
  The tests already accept comm size 2 (`parallel_helpers.check_comm_size` defaults to
  `(1, 2, 4)`). This is a CI change, handled in a separate PR. Generation keeps producing
  `comm_sizes = [1, 2, 4]` unchanged.

## Plan

### Tier 0 — first increment

**0.1 Per-experiment versioning + task selectors.**
- `definitions.py:159` — remove the `= 6` default; `version: int` becomes required.
  Spell out `version=` on each of the six `Experiments.*` entries at its honest current
  value.
- `run_serialization.py` — add `--experiment NAME` (repeatable, default all),
  `--comm-size N` (repeatable), `--dry-run`.
- **These must ship in the same commit.** `cleanup_exp_output()` deletes and
  `copy_ser_data()` writes into `mpitask{N}_{exp}_v{VV}` derived from each experiment's
  own version. With per-experiment versions and no selector, a bulk run regenerates
  unbumped experiments into *already-published* names.

**0.2 Task isolation and a result table.**
- Anything going wrong while preparing one dataset — slurm failure, missing `ser_data`,
  unparseable log, metadata write failure — raises inside that task and stops it.
- `run_experiment` catches at its own boundary and returns a `TaskResult`
  (`experiment, comm_size, status, job_id, tar_path, error`) instead of propagating.
- `run_serialization()` collects results, prints a summary table, writes
  `<output_root>/run_summary.json`, and exits 1 if any task failed.
- ~15 lines. It is what lets the remaining 17 tasks of an 18-task campaign complete.

**0.3 `archive_metadata.json` written into every archive.**
- New helpers in `run_serialization.py`: `parse_icon_log_banner()`, `harvest_git()`,
  `write_archive_metadata()`.
- Call site: `run_experiment`, between `copy_ser_data()` and `tar_folder()`, so it lands
  as a top-level tar member next to `ser_data/` — where
  `read_experiment_config_from_fortran` already reads `icon_master.namelist.json` from.
- Top-level sections so the file can grow: `archive`, `provenance`, `content`.

```json
{
  "schema": "icon4py-archive-metadata/1",
  "archive": {
    "experiment": "exclaim_ch_r04b09_dsl", "version": 7, "comm_size": 1,
    "filename": "mpitask1_exclaim_ch_r04b09_dsl_v07.tar.gz",
    "generated_at": "2026-07-30T09:14:22Z"
  },
  "provenance": {
    "icon": {
      "sha": "bebe90513b1da6842d0609579343af57b63a8158",
      "describe": "icon-2025.10-dwd-2.0-331-gbebe90513b1da...",
      "repository": "git@gitlab.dkrz.de:icon/icon-nwp.git",
      "branch": "serialize_tmx_sfc",
      "version": "2026.04",
      "build_date": "20260706", "build_time": "135858",
      "executable": "/capstor/.../build_serialize/bin/icon",
      "externals": {"icon-land": "icon-land-2026.04-18-geb24d743...",
                    "ecrad": "ecrad-safeguard-09666303-13-g19ceb48...",
                    "rte-rrtmgp": "v1.7-18-g0a8f3011a..."},
      "source": "LOG.exp.exclaim_ch_r04b09_dsl_sb.run.983972.o"
    },
    "icon4py": {"sha": "8f081b248...", "branch": "physics_driver_l2", "dirty": false},
    "runtime": {"slurm_job_id": "983972", "uenv": "icon/25.2:v3",
                "partition": "normal", "account": "cwd01", "ntasks": 1,
                "host": "nid005281"}
  },
  "content": {
    "static_record_count": 198, "field_record_count": 7071,
    "static_fingerprint": "sha256:1f3a9c...",
    "ranks": 1
  }
}
```

**0.4 `serdata fingerprint` / `serdata diff`, report-only.**
New `scripts/python/serdata.py`, typer group, stdlib only.
- `fingerprint`: join `MetaData.savepoint_vector.fields_per_savepoint[i][sp][field] = occ`
  against `ArchiveMetaData.fields_table[field][occ][1]`, **all ranks**, key
  `(savepoint_name, ordinal_within_name, field)`, `zfill(64)` before comparing.
- `diff OLD NEW [--report FILE.md]`: baseline is either `--baseline DIR` or the previous
  version's directory under `output_root` / `ICON4PY_TEST_DATA_PATH`. If neither exists,
  the report header says `BASELINE: none — no comparison performed`.
- **Namelist diff comes free** (fact 4): compare `NAMELIST_ICON_output_atm.json` between
  the two versions and report every changed key. This catches upstream default drift and
  a changed grid/extpar file with no archive change at all, and it works on every archive
  generated to date.
- Wired into `run_serialization` after `tar_folder`; result attached to `TaskResult` and
  rendered in the 0.2 table. The verdict is a string, not an exception, not an exit code —
  it must not abort a campaign.
- Report goes to stdout and `<output_root>/reports/`, **not** into the tar: it is fully
  regenerable from two archives, and nobody opens a markdown file inside an 11 GB tarball.

Rendered for the incident:

```
exclaim_ch_r04b09_dsl  v05 -> v06   VERDICT: REVIEW
provenance : icon 818095390c -> 4b1d02c7f | uenv icon/25.2:v3 UNCHANGED
namelists  : 0 changed keys
structure  : savepoints +0 -0 | fields +0 -0 | dims changed 0
STATIC          6/198 changed  <-- must be justified
  icon-grid            primal_normal_cell_x/y, dual_normal_cell_x/y
  interpolation-state  pos_on_tplane_e_x/y
INITIAL-STATE   0/25 changed
EVOLVING        first divergence: solve-nonhydro-init#1 (z_rho_e) | 3893/6633 records
upstream (geometry-scoped): 10 commits
  a9435ae531 [nwp] Improve wave energy propagation near the coast
      src/shr_horizontal/mo_intp_coeffs.f90 | 317 +++---
```

Git range: `git log --oneline --no-merges --left-right OLD...NEW -- <paths>`, right side
only. Three-dot, not two: the serialization branches are divergent siblings, and a later
build can sit on an older upstream.

**0.5 `serdata backfill <testdata_root>`.**

*What it is for.* The diff in 0.4 needs a predecessor. Today's archives contain Serialbox
hashes (fact 2) and an ICON log (fact 1), but no `archive_metadata.json` — that only starts
appearing with archives generated after 0.3 lands. Without backfill, the first regeneration
under the new scheme has nothing to compare against, and the tooling only becomes useful on
the *second* one, months later.

*What it does.* Walks every extracted `mpitask*_*_v*/` under a testdata root; for each, parses
the log that is already sitting there, writes `archive_metadata.json` into that local
directory, and computes the fingerprint. Purely local: it does not re-tar, does not
re-upload, does not touch the published archives, and never contacts Santis. Runs offline
on a laptop against `~/projects/icon4py/testdata/ser_icondata/`.

*Effect.* The next regeneration can diff against v05/v06/v07 on day one. It is also what
would have reduced this week's archaeology to a five-minute job.

**0.6 Fixture tests.**
`scripts/tests/test_serdata.py` with committed fixtures: one real log, and a *trimmed*
`MetaData` / `ArchiveMetaData` pair (static savepoints only, a few KB). Cover banner parse,
banner-parse failure, the `fields_table` join including deduplicated records, savepoint
classification, the floor assertion, namelist diff, and report rendering. Without these,
several hundred lines of new code would first execute during a rare, high-pressure
regeneration.

**0.7 Print the next commands.**
On success, print the literal upload command, the exact `definitions.py` edit, and a
paste-ready PR body containing the report. The six-month recall gap is the real adversary.

### Tier 1 — after Tier 0 has survived one real regeneration

**1.1 Pre-upload verification sweep** — the strongest gate.
Layouts are byte-identical by construction (`OUTPUT_ROOT` uses the same
`get_ranked_experiment_name_with_version` helper the consumer uses), so pointing pytest at
the fresh tree needs no new concepts:

```bash
ICON4PY_TEST_DATA_PATH=<build_serialize/experiments> \
ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0 \
uv run --group test --frozen pytest -n0 --datatest-only --backend=gtfn_cpu model/common
```

≈11 min at the CI reference; all three incident fields are asserted within it. Exposed as
`run_serialization --run-tests` (opt-in) once a working icon4py env and GT4Py cache exist on
Santis. Until then it is a documented manual step in the runbook.

**1.2 `known_hash` on download.**
`data_handling.py:66` — take the digest from a committed `model/testing/testdata/registry.txt`,
tolerating a missing entry with a warning. Catches silent re-upload, partial upload, and any
archive that was not produced by the script.

**1.3 Committed lockfile + CI check** — the only blocking gate in the plan.

*Clarification*: the lockfile stores **hashes of the `static` and `initial-state` savepoint
fields**. It does not put grid data or initial-state data into archives — both already live
in `ser_data/`. It records what those fields hashed to when the version was blessed, so that
a later change to the published archive, or a mistaken version bump, is caught in CI rather
than by a mystery test failure.

- `serdata bless <experiment> --version N` → `model/testing/testdata/<experiment>.lock.json`:
  per `(comm_size, rank)`, `icon_sha`, `uenv`, `tar_sha256`, and the `static` +
  `initial-state` field hashes truncated to 16 hex (~20 KB per archive). Not `evolving`.
- `model/testing/tests/testing/datatest/test_testdata_lock.py`, datatest-marked, recomputes
  and compares. **Parametrise on experiment only** — `comm_size` comes from
  `process_props.comm_size`, so a test cannot choose it.
- Skip-with-warning for archives predating the scheme. Advisory for one cycle, then strict.
- This blocks a PR, not a slurm batch. That is the only correct place to block.

**1.4 Advisory consumer surfacing.**
A `@functools.cache`d metadata loader called from `datatest_utils.download_experiment`
(memoisation is mandatory — `download_ser_data` is function-scoped), accumulating into a
module dict rendered from the existing `pytest_terminal_summary` (`pytest_hooks.py:246`).
**Warning only, never `raise`**: a mismatch is far more likely to mean a stale local copy
than an object-store overwrite, and a hard failure turns that into a CI outage with a false
diagnosis. `pytest_report_header` cannot work — downloads are lazy and function-scoped.

## Consequences

**Easier**: attributing any future data delta to an upstream commit; regenerating one
experiment without invalidating five others; recovering a campaign after one task fails;
verifying downloaded data is what was published.

**Harder**: the regeneration runbook gains steps (mitigated by 0.7 printing them); `serdata`
becomes a second thing to maintain; the static-class floor constants need updating when
savepoints are legitimately added.

**Follow-up decision required**: whether the `a9435ae531` delta is a regression or an
improvement. The boundary row went from an artificial zero to a real value, so the current
test slices may be encoding a Fortran limitation that no longer exists. Recording this
explicitly sets the norm: when the static class fires on a legitimate upstream improvement,
the cheap path must be "bless it with a one-line rationale", not "reach for `--allow-drift`".

## Alternatives considered

### Pin the upstream ICON commit (rejected)

- Good, because it would prevent drift rather than detect it.
- Bad, because `settings.iconf90_repo_dir` is used in exactly one place (copying
  `run/exp.<name>_sb`) and has **no verified causal link** to `build_serialize/bin/icon`;
  every preflight check would validate the wrong tree.
- Bad, because it requires rebase-not-merge discipline on a long-lived private branch, and
  the serialization branches are divergent siblings.
- Bad, because it gives zero data-side signal: a pool grid swap or a uenv republish yields a
  clean pin and a confident all-clear on corrupted data.
- The one useful element — the scoped upstream-commit report — is kept inside `serdata diff`.

### Archiving `nml.atmo.log` and `run/exp.<name>_sb` (rejected)

- Bad, because both are redundant: `NAMELIST_ICON_output_atm(.json)` already carries every
  namelist value including defaults, and the resolved grid/extpar paths are already in the
  archived namelists (fact 4).
- Replaced by diffing files already present in every archive.

### Hard consumer-side failure on metadata mismatch (rejected)

- Bad, because it cannot fire for any archive published to date: `.extraction_complete`
  short-circuits before the lock, so existing checkouts never gain the file.
- Bad, because its likeliest real trigger is human bookkeeping, not corruption.
- Kept as a warning (1.4).

### Storing the baseline sha in `definitions.py` (rejected)

- Bad, because it requires editing `version=` *before* generation and `icon_revision=`
  *after*, in two passes. Doing both at once collapses the diff range into a false all-clear;
  forgetting the second yields a hard CI failure with a false diagnosis.
- Bad, because the generation script reads the *vendored* icon4py under icon-exclaim, not the
  development checkout, so the value would be maintained by hand in two diverging places.
- Replaced by recomputing baselines from `ser_data/`.

### Raising drift out of a worker thread (rejected)

- Bad, because it propagates through `future.result()` and aborts the remaining comm sizes
  for all experiments after hours of slurm queue.
- Bad, because it guarantees an `--allow-drift` escape hatch becomes permanent on second use.
- Task-local failures still raise and stop that task (0.2). Drift is not a task failure.

### Reading `.dat` values directly via numpy memmap (rejected)

- Bad, because it reimplements Serialbox's reader; `field_map` carries
  `__i/j/k minus/plus halosize` padding a naive memmap ignores. It fails by returning
  plausible wrong numbers, on the day you trust them. Use `Serializer.read()` or inspect
  manually.

### Committing generated `reports/*.md` (rejected)

- Bad, because generated prose rots in a source repo and gets rubber-stamped in review.
- Replaced by: lockfile diff in the repo, report in the PR body.

## Documentation work

Move the HackMD HOWTO in-repo to `docs/testdata_generation.md`, leaving a HackMD pointer.

Corrections required:
- Publication target is `https://rgw.cscs.ch/c2sm:testdata` with layout
  `experiments/{mpitaskN_name_vNN}.tar.gz`, `grids/{name}.tar.gz`,
  `muphys/{type}/{name}.tar.gz` — the body still describes Polybox.
- Version scheme becomes per-experiment; the rule is **never re-upload an existing object name**.
- Paths: `SerializationSettings.defaults()` asserts the icon4py checkout is literally named
  `icon4py` and derives `ROOT_PROJECT_DIR` from `__file__.parents[2]`;
  `build_dir = ROOT/build_serialize`; `OUTPUT_ROOT = build_serialize/experiments/ser_icondata`.
  State explicitly that the ICON source driving the binary is the icon-exclaim in-tree build,
  and that `ROOT/icon` is used only to copy `run/exp.<name>_sb`.
- Slurm/uenv facts are hardcoded at `run_serialization.py:84-88`: account `cwd01`, partition
  `normal`, 15 min, uenv `icon/25.2:v3`.
- Invocation is `./scripts/run run-serialization`, plus the new selectors.
- `version = 3` is wrong.

New content required:
- The serialization-branch situation on `git@gitlab.dkrz.de:icon/icon-nwp.git`: branches get
  rebased and deleted, so **push a tag for anything you generate from**. Instrumentation lives
  in `src/serialization/mo_icon4py_verification.f90`.
- The end-to-end runbook: bump one experiment → `--dry-run` → generate → read the `serdata diff`
  report → run the datatest sweep against the fresh tree → upload → `serdata bless` → PR with
  version bump + lockfile + report in the body.
- Archive contents (`ser_data/`, namelists and their `.json`, `LOG.*.o`,
  `archive_metadata.json`) and the rule that additive files are safe while renaming or nesting
  is not.
- How to read `archive_metadata.json` and turn two shas into
  `git log --left-right OLD...NEW -- src/shr_horizontal src/grid src/parallel_infrastructure`.
- The consumer-side one-liner (`ICON4PY_TEST_DATA_PATH`, `ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0`)
  and the `.extraction_complete` wipe footgun.

## References

- ICON commit `a9435ae531`, icon-nwp MR 2138, *"[nwp] Improve wave energy propagation near the coast"*
- `src/shr_horizontal/mo_intp_coeffs.f90` — `complete_patchinfo`, `calculate_tangent_plane_at_edge`
- `src/shared/mo_util_vcs.f90:169` — `show_version`
- HackMD `zHbtiaa2R4mFEA1RT3M8_A` — current HOWTO
