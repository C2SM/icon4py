# Generating serialized test data

ICON4Py's datatests validate against Serialbox dumps produced by an instrumented ICON
Fortran build. This is the runbook for regenerating them.

Generation runs on CSCS Santis under slurm. The recipes in
[Which ICON build produced an archive](#which-icon-build-produced-an-archive) work anywhere.

## What an archive is

One archive per `(experiment, communicator size)`, named
`mpitask{N}_{experiment}_v{VV}.tar.gz`, published to `https://rgw.cscs.ch/c2sm:testdata`
under `experiments/` (grids live under `grids/`, muphys data under `muphys/{type}/`).

Each contains `ser_data/` (the Serialbox dump), the input and post-read namelists with
their `.json` equivalents, and `LOG.exp.{experiment}_sb.run.{jobid}.o` — the slurm log,
which carries ICON's startup banner and therefore the identity of the build that produced
the data.

### Versions are per experiment

`ExperimentDescription.version` in
`model/testing/src/icon4py/model/testing/definitions.py` has no default: each experiment
carries its own version and is regenerated on its own. A shared default meant that
regenerating one experiment invalidated the published archives of the other five.

**Never re-upload an existing object name.** Downloads are cached behind an
`.extraction_complete` marker, so a checkout that already extracted a version will never
notice that it changed. Bump the version before regenerating, and use `--dry-run` to check
which names a campaign is about to write.

## Which ICON build produced an archive

Every archive ever published answers this, with no tooling:

```bash
grep -m1 '^ revision:' <archive dir>/LOG.*.o
#  revision: icon-2026.04-dwd-1.9-136-g1df503335b8726dc445659c21020f4f09d9acc3d
```

Datatest runs print the same thing for every archive they touch, in the terminal summary:

```
 Serialized test data
  mpitask1_exclaim_ch_r04b09_dsl_v06           icon-2026.04-dwd-1.9-136-g1df50333...
```

### Explaining a reference value that changed

That string is a `git describe`, and **git accepts it as a revision**, so it can be used
in a range directly. When a datatest starts disagreeing with the reference data:

```bash
# 1. the revisions of the two archives (the old one may need re-downloading)
OLD=$(grep -m1 '^ revision:' <old archive>/LOG.*.o | cut -d' ' -f3)
NEW=$(grep -m1 '^ revision:' <new archive>/LOG.*.o | cut -d' ' -f3)

# 2. which Fortran sources compute the field the test names
grep -rl primal_normal_cell <icon checkout>/src/shr_horizontal

# 3. what changed in them between the two builds
git -C <icon checkout> log --oneline --no-merges "$OLD".."$NEW" -- src/shr_horizontal/mo_intp_coeffs.f90
```

Scope by the file the field comes from, not by the subsystem: a commit that moved three
geometry fields was titled *"[nwp] Improve wave energy propagation near the coast"*, so the
subject line is no help but the path is decisive.

If the range comes out empty, check the release lineage in the two strings — a jump like
`icon-2025.10-dwd-2.0-213` → `icon-2026.04-dwd-1.9-136` means the builds are on different
release branches rather than a few commits apart.

## Setup on Santis

The build lives in the `icon-exclaim` tree, which vendors `icon` and `icon4py` alongside
`build_serialize/`.

> **Verify before relying on this section.** These commands come from the previous HackMD
> HOWTO and have not been re-run while writing this. Correct them in place next time.

```bash
git clone git@github.com:C2SM/icon-exclaim.git icon-exclaim.serialize
cd icon-exclaim.serialize && git checkout icon4py-dev
uenv start --view=default icon/25.2:v3
./install_dependencies.sh --icon git@gitlab.dkrz.de:icon/icon-nwp.git#<branch> --icon4py <branch>
./setup.sh build_serialize
```

Instrumentation lives in `icon/src/serialization/mo_icon4py_verification.f90`, enabled per
experiment through `icon4py_verification_nml`.

Serialization branches on `icon-nwp` get rebased and deleted — **push a tag for whatever
you generate from**, or the revision recorded in the archives will eventually point at
nothing.

## Generating

Which experiments and communicator sizes run is set in the
`START DEFAULT USER CONFIGURATION` block of `scripts/python/run_serialization.py`, along
with the slurm account, partition and uenv.

```bash
cd icon-exclaim.serialize/icon4py
source .venv/bin/activate

./scripts/run run-serialization --dry-run    # names it would write, then stop
./scripts/run run-serialization
```

The run refuses to start from a modified icon4py checkout (`--allow-dirty` to override) and
prints the ICON source tree's `git describe` before submitting anything, so an unexpected
upstream revision can still be reverted rather than discovered hours later.

A failing task no longer aborts the campaign: each `(experiment, comm_size)` is reported on
its own and the command exits non-zero at the end if any failed.

When generation finishes, the datatests run against the fresh tree before anything is
published (`--skip-tests` to skip). That is the check that answers whether the new data
actually breaks ICON4Py. To run it by hand:

```bash
ICON4PY_TEST_DATA_PATH=<...>/build_serialize/experiments ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0 \
  uv run --group test --frozen pytest --datatest-only --backend=gtfn_cpu model/common
```

## Publishing

From the directory holding the tarballs:

```bash
aws --profile cscs-icon4py s3 sync . s3://testdata/experiments/ \
  --exclude "*" --include "*.tar.gz"
```

Then open a PR with the version bump in `definitions.py`.

## Using data that is not published yet

```bash
export ICON4PY_TEST_DATA_PATH=/path/to/testdata
export ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0
```

**Footgun:** with downloads enabled, the framework deletes the contents of a target
directory before extracting. If you place an archive by hand, `touch .extraction_complete`
inside it first, or disable downloads.
