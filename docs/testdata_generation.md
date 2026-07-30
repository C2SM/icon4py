# Generating serialized test data

icon4py's datatests validate against Serialbox dumps produced by an instrumented ICON
Fortran build. This document is the runbook for regenerating them.

Generation runs on CSCS Santis under slurm; nothing here works on a laptop except the
inspection commands in [Reading an archive](#reading-an-archive).

## What an archive is

One archive per `(experiment, communicator size)`, named
`mpitask{N}_{experiment}_v{VV}.tar.gz` and published to `https://rgw.cscs.ch/c2sm:testdata`
under:

| prefix           | contents                               |
| ---------------- | -------------------------------------- |
| `experiments/`   | `mpitask{N}_{experiment}_v{VV}.tar.gz` |
| `grids/`         | `{grid}.tar.gz`                        |
| `muphys/{type}/` | `{name}.tar.gz`                        |

Each experiment archive contains:

| entry                                               | what it is                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `ser_data/`                                         | the Serialbox dump: one `.dat` per field, plus `MetaData-*.json` and `ArchiveMetaData-*.json` per rank |
| `NAMELIST_{experiment}_sb`, `NAMELIST_expname.json` | the input namelist: only what the experiment sets explicitly                                           |
| `NAMELIST_ICON_output_atm`, `.json`                 | ICON's post-read dump: every variable at its effective value, **defaults included**                    |
| `icon_master.namelist`, `.json`                     | the master namelist                                                                                    |
| `LOG.exp.{experiment}_sb.run.{jobid}.o`             | the slurm log, including ICON's startup version banner                                                 |
| `archive_metadata.json`                             | machine-readable identity of the archive (see below)                                                   |

Adding files to an archive is safe — the consumer reads by name. Renaming or nesting
them is not.

### Versions are per experiment

`ExperimentDescription.version` in
`model/testing/src/icon4py/model/testing/definitions.py` has **no default**: each
experiment carries its own version and is regenerated on its own. This is deliberate. A
shared default meant that regenerating one experiment invalidated the published archives
of all the others.

Two rules follow:

- **Never re-upload an existing object name.** Downloads are cached behind an
  `.extraction_complete` marker, so a checkout that already extracted a version will
  never see that it changed.
- **Bump the version of an experiment before regenerating it**, and pass `--experiment`
  so that experiments you did not bump are not rebuilt into their published names.

## Setup on Santis

The build lives in the `icon-exclaim` tree, which vendors both `icon` and `icon4py` as
top-level directories:

```
icon-exclaim.serialize/
  icon/                 # the Fortran model
  icon4py/              # this repository
  build_serialize/
    run/                # runscripts
    experiments/        # experiment output, and ser_icondata/ below it
```

> **Verify before relying on this section.** The commands below come from the previous
> HackMD HOWTO and have not been re-run while writing this document. Correct them in
> place the next time you generate data.

```bash
git clone git@github.com:C2SM/icon-exclaim.git icon-exclaim.serialize
cd icon-exclaim.serialize
git checkout icon4py-dev
uenv start --view=default icon/25.2:v3
./install_dependencies.sh --icon git@gitlab.dkrz.de:icon/icon-nwp.git#<branch> --icon4py <branch>
./setup.sh build_serialize
```

The serialization instrumentation is in
`icon/src/serialization/mo_icon4py_verification.f90`, enabled per experiment through
`icon4py_verification_nml`:

```fortran
&icon4py_verification_nml
    dynamics = .TRUE.
    static = .TRUE.
    physics = .FALSE.
    serialization_start_date = "2021-06-20T12:00:00Z"
    serialization_end_date = "2021-06-20T12:10:00Z"
/
```

### Pin the ICON branch you generate from

Serialization branches on `git@gitlab.dkrz.de:icon/icon-nwp.git` get rebased and deleted.
**Push a tag for whatever commit you generate from**, otherwise the revision recorded in
`archive_metadata.json` will eventually point at nothing.

This matters: in June 2026 a regeneration silently picked up `a9435ae531`
("[nwp] Improve wave energy propagation near the coast"), which moved three geometry
fields from `rl_start=2` to `rl_start=1` in `src/shr_horizontal/mo_intp_coeffs.f90` and
broke three icon4py datatests with no recorded provenance to explain it. That incident is
what the comparison step below exists to catch.

## Generating

Settings — slurm account `cwd01`, partition `normal`, 15 minutes, uenv `icon/25.2:v3`,
communicator sizes `[1, 2, 4]` — are hardcoded in `SerializationSettings.defaults()` in
`scripts/python/run_serialization.py`. Change them there.

```bash
cd icon-exclaim.serialize/icon4py
source .venv/bin/activate

# what would run, without submitting anything
./scripts/run run-serialization --dry-run

# one experiment, all communicator sizes
./scripts/run run-serialization --experiment exclaim_ch_r04b09_dsl

# a subset
./scripts/run run-serialization -e exclaim_gauss3d -c 1 -c 4
```

Output lands in `build_serialize/experiments/ser_icondata/`.

A failing task no longer aborts the campaign: each `(experiment, comm_size)` task is
reported on its own, the run ends with a summary table and `run_summary.json`, and the
command exits non-zero if anything failed.

## Reviewing before publishing

Every task compares its archive against the previous version and writes a report to
`ser_icondata/reports/`. Read it before uploading anything.

```
exclaim_ch_r04b09_dsl  ..._v05 -> ..._v06   VERDICT: REVIEW
provenance : icon 818095390c -> 4b1d02c7f
namelists  : 0 changed, 0 added, 0 removed
structure  : savepoints +0 -0 | fields +0 -0
STATIC         167 unchanged, 6 changed, 0 added, 0 removed
  changed icon-grid              primal_normal_cell_x, primal_normal_cell_y, ...
  changed interpolation-state    pos_on_tplane_e_x, pos_on_tplane_e_y
INITIAL-STATE  25 unchanged, 0 changed, 0 added, 0 removed
EVOLVING       ...
```

Savepoints fall into three classes, judged by different standards:

| class           | savepoints                                                                              | standard                                                                           |
| --------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `static`        | `icon-grid`, `metric-state`, `interpolation-state`, `smooth-topo-savepoint`, `tmx-init` | every changed field is named and must be justified                                 |
| `initial-state` | `prognostics`, `diagnostics` at `location=initial-state`                                | as above, but separated so an intentional IC change does not read as a grid change |
| `evolving`      | everything written per timestep                                                         | counted only                                                                       |

The verdict is `REVIEW` when a static or initial-state record **changed or disappeared**,
or when a new savepoint is not yet classified. Records merely *appearing* in a guarded
class are normal — that is what adding instrumentation looks like.

`UNVERIFIED` means the previous version was not on disk, so nothing was compared. Extract
it and re-run the comparison by hand:

```bash
./scripts/run serdata diff <new archive dir> --baseline <previous archive dir>
```

If a new savepoint is reported as `UNCLASSIFIED`, add it to `STATIC_SAVEPOINTS` in
`scripts/python/serdata.py` or leave it evolving — but decide, rather than letting it
default into the unguarded class.

### Run the datatests before uploading

This is the step that catches what the fingerprint cannot: whether the delta actually
breaks icon4py. The generated layout is identical to the consumed one, so pytest can be
pointed straight at it:

```bash
ICON4PY_TEST_DATA_PATH=<...>/build_serialize/experiments \
ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0 \
  uv run --group test --frozen pytest -n0 --datatest-only --backend=gtfn_cpu model/common
```

## Publishing

> **Fill this in.** The upload command for `rgw.cscs.ch/c2sm:testdata` is not recorded
> anywhere in the repository and is not reproduced here rather than guessed. Write the
> exact command down the next time you publish.

Then open a PR containing the version bump in `definitions.py` and the comparison report
in the PR body.

## Reading an archive

These run offline, on a laptop, against an extracted archive:

```bash
# what the archive contains, by savepoint class
./scripts/run serdata fingerprint ~/icon4py/testdata/ser_icondata/mpitask1_exclaim_gauss3d_v05

# compare two versions
./scripts/run serdata diff <new> --baseline <old>

# reconstruct archive_metadata.json for archives generated before it existed
./scripts/run serdata backfill ~/icon4py/testdata/ser_icondata
```

`backfill` reads the ICON log each archive already carries. It writes only into the local
extracted directories — it does not re-tar, re-upload, or contact Santis. Run it once so
that the next regeneration has a baseline to compare against.

### `archive_metadata.json`

```json
{
  "schema": "icon4py-archive-metadata/1",
  "archive": {"experiment": "...", "version": 6, "comm_size": 1, "filename": "...", "generated_at": "..."},
  "provenance": {
    "icon": {"sha": "...", "describe": "...", "repository": "...", "branch": "...", "externals": {...}},
    "icon4py": {"sha": "...", "branch": "...", "dirty": false},
    "runtime": {"slurm_job_id": "...", "uenv": "...", "partition": "...", "account": "..."}
  }
}
```

The ICON revision comes from the version banner in the slurm log, which is the only
artifact with a verified link to the binary that produced the data. An `archive` section
with `"backfilled": true` was reconstructed after the fact and has no runtime section.

To find out what moved upstream between two archives:

```bash
git -C <icon checkout> log --oneline --no-merges --left-right <old sha>...<new sha> \
  -- src/shr_horizontal src/grid src/parallel_infrastructure
```

Use three dots and read the right-hand side: serialization branches are divergent
siblings, so a later build can sit on an older upstream, and two-dot ranges silently come
out empty.

## Using data that is not published yet

```bash
export ICON4PY_TEST_DATA_PATH=/path/to/testdata
export ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0
```

**Footgun:** with downloads enabled, the framework deletes the contents of a target
directory before extracting. If you place an archive by hand, `touch .extraction_complete` inside it first, or disable downloads.
