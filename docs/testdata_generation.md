# Generating serialized test data

ICON4Py's datatests validate against Serialbox dumps produced by an instrumented ICON
Fortran build, one archive per `(experiment, communicator size)`, published to
`https://rgw.cscs.ch/c2sm:testdata` under `experiments/`. Generation runs on CSCS Santis.

Each archive carries `ser_data/`, the namelists, and the slurm log — which contains ICON's
startup banner, and therefore the identity of the build that produced the data.

## Which ICON build produced an archive

```bash
grep -m1 '^ revision:' <archive dir>/LOG.*.o
#  revision: icon-2026.04-dwd-1.9-136-g1df503335b8726dc445659c21020f4f09d9acc3d
```

Datatest runs print the same for every archive they touch, in the terminal summary.

That string is a `git describe`, and git accepts it as a revision. So when a datatest starts
disagreeing with its reference data:

```bash
OLD=$(grep -m1 '^ revision:' <old archive>/LOG.*.o | cut -d' ' -f3)
NEW=$(grep -m1 '^ revision:' <new archive>/LOG.*.o | cut -d' ' -f3)

grep -rl primal_normal_cell <icon checkout>/src/shr_horizontal   # where the field comes from
git -C <icon checkout> log --oneline --no-merges "$OLD".."$NEW" -- src/shr_horizontal/mo_intp_coeffs.f90
```

Scope by the file the failing test's field comes from, not by subsystem: the commit that
last moved three geometry fields was titled *"[nwp] Improve wave energy propagation near the
coast"*, so only the path is decisive.

## Versions

`ExperimentDescription.version` in `model/testing/.../definitions.py` has no default: each
experiment is versioned and regenerated on its own.

**Never re-upload an existing object name.** Downloads are cached behind an
`.extraction_complete` marker, so a checkout that already extracted a version will never
notice it changed. Bump the version first, and check with `--dry-run`.

## Generating

Experiments, communicator sizes, slurm account and uenv are set in the
`START DEFAULT USER CONFIGURATION` block of `scripts/python/run_serialization.py`.

```bash
cd icon-exclaim.serialize/icon4py && source .venv/bin/activate
./scripts/run run-serialization --dry-run
./scripts/run run-serialization
```

The run prints the ICON source tree's `git describe` before submitting anything, and refuses
to start from a modified icon4py checkout (`--allow-dirty` overrides). A failing task no
longer aborts the campaign.

Before publishing, run the datatests against the new data:

```bash
ICON4PY_TEST_DATA_PATH=<...>/build_serialize/experiments ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0 \
  uv run --group test --frozen pytest --datatest-only --backend=gtfn_cpu model/common
```

Then upload, and open a PR with the version bump:

```bash
aws --profile cscs-icon4py s3 sync . s3://testdata/experiments/ --exclude "*" --include "*.tar.gz"
```

## Building on Santis

*Carried over from the previous HackMD page and not re-run since; correct in place.*

```bash
git clone git@github.com:C2SM/icon-exclaim.git icon-exclaim.serialize
cd icon-exclaim.serialize && git checkout icon4py-dev
uenv start --view=default icon/25.2:v3
./install_dependencies.sh --icon git@gitlab.dkrz.de:icon/icon-nwp.git#<branch> --icon4py <branch>
./setup.sh build_serialize
```

Instrumentation lives in `icon/src/serialization/mo_icon4py_verification.f90`. Serialization
branches on `icon-nwp` get rebased and deleted, so **push a tag for whatever you generate
from** or the recorded revision will eventually point at nothing.

## Using data that is not published yet

```bash
export ICON4PY_TEST_DATA_PATH=/path/to/testdata
export ICON4PY_ENABLE_TESTDATA_DOWNLOAD=0
```

**Footgun:** with downloads enabled the framework deletes the target directory's contents
before extracting. If you place an archive by hand, `touch .extraction_complete` inside it
first, or disable downloads.
