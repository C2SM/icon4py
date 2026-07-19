# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed reference-data reader for the M0 binary dumps written by the
instrumented `scale_atmos_phy_mp_amps.F90` (scale_amps repo, branch
`cloudlab_port`, `AMPS_DUMP_micro`/`AMPS_DUMP_sed` subroutines).

This is a TYPED PORT of `scale_amps/scripts/amps_dump_reader.py` (record
parsing, rank-aware keys, endian auto-detect) -- that script stays the
authoritative implementation for cluster-side dump -> npz conversion
(`aggregate()` + `np.savez_compressed`, run on the cluster where the raw
`.bin` dumps live); this module is the icon4py-side counterpart, reading
BOTH the raw `.bin` dumps directly (`read_dump_file`/`load_reference` on a
directory) AND the already-converted `.npz` archive
(`load_reference` on a `.npz` file) into typed `MicroRecord`/`SedRecord`
dataclasses instead of untyped dicts. Keep the record layout (magic
numbers, field order, byte-order auto-detection) in sync with
`scale_amps/scripts/amps_dump_reader.py` if either ever changes -- both
read the SAME binary format, written by the SAME Fortran subroutines.

Binary layout v1 (unchanged from the scale_amps reader): a sequence of
records; ints int32, reals float64, byte order auto-detected per file
(little-endian by default; some cluster builds force big-endian
unformatted/stream I/O, e.g. gnu Makedefs' `-fconvert=big-endian` or
intel's `-convert big_endian` -- this applies to stream writes too), arrays
prefixed by one int32 size per rank, Fortran (column-major) order. Record
types: micro (magic `1095586131`) and sed (magic `1095586132`) -- field
order defined in the M0 plan Tasks 2-3 and mirrored in the module-level
`MICRO_*`/`SED_*` tuples below, exactly as scale_amps's reader has them.

Every rank writes its own files, named `amps_dump_r{RRRRRR}_t{TTT}.bin`
(rank, thread); every rank dumps the same LOCAL i/j box, so every typed
record carries its own `rank` (see `read_dump_file`'s `rank` parameter and
`MicroRecord.rank`/`SedRecord.rank`) and `RefDataset.micro_pairs()`/
`.sed_pairs()` key by `(rank, TIME_AMPS, i, j[, isn])` -- omitting rank
would silently collide records from different ranks that dump the
identical LOCAL (i, j) box. Thread is deliberately NOT part of the pairing
key: within one rank, a given (i, j) column is processed by exactly one
thread, so `(rank, TIME_AMPS, i, j)` is already unique without it.

Version fields are validated with `raise ValueError` (not `assert`) so the
check survives `python -O` (carry-forward #2 from the M0 plan).

-------------------------------------------------------------------------
Sed-input derivation notes (M0 final review Minor 5)
-------------------------------------------------------------------------

`sclsedprz_original`'s sedimentation call
(`scale_atmos_phy_mp_amps.F90:2417-2420`, liquid; `:2471-2474`, ice --
identical argument list both times) takes six scalar/array inputs that
`AMPS_DUMP_sed` (`scale_atmos_phy_mp_amps.F90:5324-5372`; its own argument
list is exactly `SED_HEADER + SED_R1` below, confirmed field-for-field
against the dump call sites at lines 2403-2410/2439-2446) does NOT capture
directly -- because they are either module-scope constants or column
diagnostics computed from other quantities, never dumped as their own
array. Whoever wires up the M2 sed replay from a `SedRecord` needs to
reconstruct these six; every line number below was read directly out of
`scale_atmos_phy_mp_amps.F90` for this task, not guessed or carried over
from a prior doc:

* `waccv = -1.0` -- a scalar CONSTANT, set ONCE outside the i/j loop
  (line 1433: `waccv = -1.0e+0 ! negative one, so that the velocity is
  negative downwards`). Not derived from any dumped field.
* `spdsfcv = 0.0` -- a scalar CONSTANT, set per-column right after the
  thermo block (line 1842; the restart-path copy of the same driver code
  repeats it verbatim at line 3361). Not derived from any dumped field.
* `momv = momz_col` -- direct copy, bit-identical, no arithmetic. In
  `Z_LOOP_01`, `momv(k) = MOMZ(k,i,j)` (line 1666) is literally the same
  array element `AMPS_DUMP_sed` passes as its `momz_col` argument
  (`MOMZ(KS-1:KE,i,j)`, call sites line 2406/2442, dump signature line
  5327/5358) -- so `momv = sed_record.momz_col`, exactly.
* `thskinv` (from `thetav`/`qvv`) -- `thskinv = thv(KS-1)` (line 1841),
  but `thv` itself is NOT dumped (only `thetav`/`qvv` are, both in
  `SED_R1`). Invert the general in-column virtual-potential-temperature
  relation (line 1661: `thetav(k) = thv(k)*(1.0+0.61*qvv(k))`) to get
  `thv(k) = thetav(k) / (1.0 + 0.61*qvv(k))`, then take the k=KS-1 entry
  (array index 0 -- every `SED_R1` column array spans `KS-1:KE`):
  `thskinv = sed_record.thetav[0] / (1.0 + 0.61*sed_record.qvv[0])`.
  CAVEAT: the Fortran's OWN literal assignment for the KS-1 level (lines
  1836-1837) does NOT go through this inversion -- it derives `thv(KS-1)`
  via the Exner relation from a hydrostatically-extrapolated `ptotv(KS-1)`
  (see `pgnd` below), and its own `thetav(KS-1)` (line 1837) divides by
  `QDRY(KS,i,j)` rather than by `1.0`, unlike every other level's formula
  (line 1661). `QDRY` is never dumped, so a bit-for-bit-exact `thskinv`
  cannot be reconstructed from `SedRecord` fields alone; the inversion
  above is the best available approximation from the dump (exact wherever
  `QDRY ~= 1`, i.e. away from heavily hydrometeor-loaded columns) --
  flagged here rather than silently assumed exact.
* `pgnd` (via Exner) -- `pgnd = 0.5*(ptotv(KS-1) + ptotv(KS))` (line
  1839). `ptotv` is NOT dumped either (only `tv`, `thetav`, `qvv`, all
  `SED_R1`). Reconstruct `ptotv` at the record's first two column indices
  (k=KS-1, k=KS -- array indices 0, 1) by inverting the SAME Exner
  relation that defines `thv`/`piv` in `Z_LOOP_01`
  (line 1644: `thv(k) = tv(k)*(PRE00/ptotv(k))**(Rdry/CPdry)`):
  `ptotv(k) = PRE00 * (tv(k)/thv(k))**(CPdry/Rdry)`, with `thv(k)` from
  the `thskinv` inversion above (`thv(k) = thetav(k)/(1+0.61*qvv(k))`).
  `Rdry=287.04`, `CPdry=1004.64` J/K/kg, `PRE00=1.0e5` Pa are the SAME
  SCALE SI constants `core/packing.py` already cites as `SCALE_RDRY`/
  `SCALE_CPDRY`/`SCALE_PRE00` (traced there to `scale_const.F90`); reuse
  those, don't re-derive. The k=KS-1 term inherits the same `QDRY`-
  availability caveat as `thskinv` above; the k=KS term (line 1638/1644)
  is the ordinary per-level formula with no such caveat.
* `dz1v = FZ(KS,i,j) - FZ(KS-1,i,j)` (line 1600) -- a first difference of
  the SAME `fz_col` array the dump captures (`fz_col = FZ(KS-1:KE,i,j)`):
  `dz1v = sed_record.fz_col[1] - sed_record.fz_col[0]`. Exact, no caveat.

None of the six derivations above is implemented as code in this module --
`ref_data.py` only READS the dump/npz; turning a `SedRecord` into
`sclsedprz_original`'s actual call arguments is M2 replay scope. This note
exists so that scope isn't re-derived from scratch later.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np


MAGIC_MICRO = 1095586131
MAGIC_SED = 1095586132
MAGIC_SETUP = 1095586133

# ---------------------------------------------------------------------------
# Record field layout (v1) -- keep in sync with scale_amps's
# scripts/amps_dump_reader.py; both parse the SAME binary format written by
# the SAME AMPS_DUMP_micro/AMPS_DUMP_sed Fortran subroutines.
# ---------------------------------------------------------------------------

MICRO_HEADER = (
    "phase",
    "TIME_AMPS",
    "i",
    "j",
    "isect",
    "nmic",
    "npr",
    "nbr",
    "ncr",
    "npi",
    "nbi",
    "nci",
    "npa",
    "nba",
    "nca",
    "mxnbin",
    "istrt",
    "jseed",
    "ifrst",
    "isect_seed",
    "nextn",
)
MICRO_R1 = ("qcvm", "v3v", "qvvm", "moist_denvm", "ptotvm", "tvm", "wbvm", "trpv_thil", "trpv_qtp")
MICRO_R4 = ("qrpvm", "qipvm", "qapvm")
MICRO_POST_EXTRA = (("dmtendlm", 3), ("dcontendlm", 3), ("dbintendlm", 4))

SED_HEADER = (
    "phase",
    "TIME_AMPS",
    "i",
    "j",
    "isect",
    "isn",
    "iadvv",
    "np",  # -> SedRecord.nprop (renamed: "np" would shadow the numpy import alias)
    "nb",
    "nc",
    "k1",
    "k2",
    "k1m",
    "k2m",
)
SED_R1 = (
    "q_this",
    "q_other",
    "qcv",
    "qtp",
    "moist_denv",
    "thetav",
    "qvv",
    "tv",
    "dens_col",
    "momz_col",
    "u_col",
    "v_col",
    "cz_col",
    "fz_col",
    "dzzmv",
    "dzvmv",
)
SED_R1_TAIL = ("den_t", "momz_t", "rhou_t", "rhov_t", "rhoe_t")


@dataclasses.dataclass(frozen=True)
class MicroRecord:
    """One `micro` (magic `MAGIC_MICRO`) dump record: a snapshot of one
    (rank, i, j) column's "compressed" (hydrometeor-bearing levels only,
    see `kmicvm`) microphysics state, at phase 1 ("pre", before the micro
    substep) or phase 2 ("post", after -- carrying the extra tendency
    diagnostics `dmtendlm`/`dcontendlm`/`dbintendlm`, `None` for phase 1).

    `rank` is NOT part of the on-disk byte stream (see module docstring);
    it is stamped by `read_dump_file`'s `rank` parameter / `load_reference`.
    """

    PHASE_PRE: ClassVar[int] = 1
    PHASE_POST: ClassVar[int] = 2

    rank: int
    phase: int
    TIME_AMPS: int
    i: int
    j: int
    isect: int
    nmic: int
    npr: int
    nbr: int
    ncr: int
    npi: int
    nbi: int
    nci: int
    npa: int
    nba: int
    nca: int
    mxnbin: int
    istrt: int
    jseed: int
    ifrst: int
    isect_seed: int
    nextn: int
    dt: float
    kmicvm: np.ndarray  # int32, (nmic,) -- Fortran-1-based k index per compressed column entry
    qcvm: np.ndarray
    v3v: np.ndarray
    qvvm: np.ndarray
    moist_denvm: np.ndarray
    ptotvm: np.ndarray
    tvm: np.ndarray
    wbvm: np.ndarray
    trpv_thil: np.ndarray
    trpv_qtp: np.ndarray
    qrpvm: np.ndarray  # (npr, nbr, ncr, nmic)
    qipvm: np.ndarray  # (npi, nbi, nci, nmic)
    qapvm: np.ndarray  # (npa, nba, nca, nmic)
    dmtendlm: np.ndarray | None = None
    dcontendlm: np.ndarray | None = None
    dbintendlm: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class SedRecord:
    """One `sed` (magic `MAGIC_SED`) dump record: a snapshot of one
    (rank, i, j, isn) sedimentation column, `isn` selecting liquid (0) or
    ice (1); phase 3 ("pre", before `sclsedprz_original`) or 4 ("post",
    after). Column arrays (`SED_R1`/`SED_R1_TAIL`) span `KS-1:KE` -- index
    0 is the extrapolated ground/surface level, index 1 is the first real
    model level.

    `rank` is NOT part of the on-disk byte stream (see module docstring);
    it is stamped by `read_dump_file`'s `rank` parameter / `load_reference`.
    """

    PHASE_PRE: ClassVar[int] = 3
    PHASE_POST: ClassVar[int] = 4

    rank: int
    phase: int
    TIME_AMPS: int
    i: int
    j: int
    isect: int
    isn: int
    iadvv: int
    nprop: int  # Fortran "np" header field -- see SED_HEADER's comment
    nb: int
    nc: int
    k1: int
    k2: int
    k1m: int
    k2m: int
    dt: float
    k1b: np.ndarray  # int32, (nb, nc)
    k2b: np.ndarray  # int32, (nb, nc)
    qpv: np.ndarray  # (nprop, nb, nc, nzh)
    q_this: np.ndarray
    q_other: np.ndarray
    qcv: np.ndarray
    qtp: np.ndarray
    moist_denv: np.ndarray
    thetav: np.ndarray
    qvv: np.ndarray
    tv: np.ndarray
    dens_col: np.ndarray
    momz_col: np.ndarray
    u_col: np.ndarray
    v_col: np.ndarray
    cz_col: np.ndarray
    fz_col: np.ndarray
    dzzmv: np.ndarray
    dzvmv: np.ndarray
    mmass: np.ndarray  # (nb, nc, nzh)
    den_t: np.ndarray
    momz_t: np.ndarray
    rhou_t: np.ndarray
    rhov_t: np.ndarray
    rhoe_t: np.ndarray
    sflx: float


Record = MicroRecord | SedRecord


# ---------------------------------------------------------------------------
# Byte-level parsing -- ported verbatim (semantics unchanged) from
# scale_amps's scripts/amps_dump_reader.py.
# ---------------------------------------------------------------------------


class _Cursor:
    def __init__(self, buf: bytes, bo: str = "<"):
        self.buf = buf
        self.pos = 0
        self.bo = bo  # "<" little-endian (default) or ">" big-endian; one
        # value per file, auto-detected from the first record's magic in
        # read_dump_file (all records in a file share the build's byte order).

    def eof(self) -> bool:
        return self.pos >= len(self.buf)

    def i4(self, n: int = 1) -> int | np.ndarray:
        out = np.frombuffer(self.buf, dtype=f"{self.bo}i4", count=n, offset=self.pos)
        self.pos += 4 * n
        return int(out[0]) if n == 1 else out.copy()

    def f8(self, n: int = 1) -> float | np.ndarray:
        out = np.frombuffer(self.buf, dtype=f"{self.bo}f8", count=n, offset=self.pos)
        self.pos += 8 * n
        return float(out[0]) if n == 1 else out.copy()

    def i1_arr(self) -> np.ndarray:
        """Despite the name (kept for fidelity with the scale_amps
        original -- the "1" means "1-D", NOT "1-byte"), this reads int32
        values: one int32 size prefix, then that many int32 elements.

        `np.atleast_1d` (NOT present in the original's `self.i4(n)`, which
        returns a bare Python `int` when `n == 1`): every dataclass field
        fed by this method (`kmicvm`, `k1b`, `k2b`) is typed `np.ndarray`,
        so a size-1 array must come back as a genuine 1-element ndarray,
        not a scalar `int`, to satisfy that type -- a real (if narrow) gap
        in the original's untyped dict-based design."""
        n = int(self.i4())
        if n <= 0:
            return np.empty(0, dtype=f"{self.bo}i4")
        return np.atleast_1d(self.i4(n))

    def rn_arr(self, ndims: int) -> np.ndarray:
        shape = tuple(int(self.i4()) for _ in range(ndims))
        n = int(np.prod(shape))
        flat = self.f8(n) if n > 0 else np.empty(0)
        return np.asarray(flat).reshape(shape, order="F")


def _detect_byte_order(buf: bytes, path: str | Path) -> str:
    """Return '<' or '>' for the file's byte order, detected from whichever
    interpretation of the first 4 bytes yields a known record magic. SCALE
    cluster builds commonly force big-endian unformatted/stream I/O
    (gnu -fconvert=big-endian, intel -convert big_endian), so this cannot be
    hardcoded. Every record in a file shares the build's byte order, so this
    is only done once, on the first magic."""
    raw = int(np.frombuffer(buf, dtype="<i4", count=1, offset=0)[0])
    if raw in (MAGIC_MICRO, MAGIC_SED):
        return "<"
    swapped = int(np.frombuffer(buf, dtype=">i4", count=1, offset=0)[0])
    if swapped in (MAGIC_MICRO, MAGIC_SED):
        return ">"
    raise ValueError(f"{path}: bad magic {raw} at byte 0")


def _read_micro(c: _Cursor, rank: int) -> MicroRecord:
    version = int(c.i4())
    if version != 1:
        raise ValueError(f"unsupported micro record version {version}")
    kwargs: dict[str, object] = {"rank": rank}
    for name in MICRO_HEADER:
        kwargs[name] = int(c.i4())
    kwargs["dt"] = float(c.f8())
    kwargs["kmicvm"] = c.i1_arr()
    for name in MICRO_R1:
        kwargs[name] = c.rn_arr(1)
    for name in MICRO_R4:
        kwargs[name] = c.rn_arr(4)
    if kwargs["phase"] == MicroRecord.PHASE_POST:
        for name, ndims in MICRO_POST_EXTRA:
            kwargs[name] = c.rn_arr(ndims)
    return MicroRecord(**kwargs)  # type: ignore[arg-type]


def _read_sed(c: _Cursor, rank: int) -> SedRecord:
    version = int(c.i4())
    if version != 1:
        raise ValueError(f"unsupported sed record version {version}")
    header: dict[str, int] = {}
    for name in SED_HEADER:
        header["nprop" if name == "np" else name] = int(c.i4())
    dt = float(c.f8())
    # k1b/k2b are written flattened (one value per (bin, class) column);
    # restore the Fortran (nb, nc) shape using the header's own nb/nc.
    k1b = c.i1_arr().reshape((header["nb"], header["nc"]), order="F")
    k2b = c.i1_arr().reshape((header["nb"], header["nc"]), order="F")
    qpv = c.rn_arr(4)
    r1 = {name: c.rn_arr(1) for name in SED_R1}
    mmass = c.rn_arr(3)
    tail = {name: c.rn_arr(1) for name in SED_R1_TAIL}
    sflx = float(c.f8())
    kwargs: dict[str, object] = {
        "rank": rank,
        "dt": dt,
        "k1b": k1b,
        "k2b": k2b,
        "qpv": qpv,
        "mmass": mmass,
        "sflx": sflx,
        **header,
        **r1,
        **tail,
    }
    return SedRecord(**kwargs)  # type: ignore[arg-type]


def read_dump_file(path: str | Path, *, rank: int = 0) -> list[Record]:
    """Parse one `amps_dump_r{rank}_t{thread}.bin` file into typed
    `MicroRecord`/`SedRecord` instances, in on-disk order.

    `rank` is NOT encoded in the byte stream itself (see module docstring:
    every rank dumps the same LOCAL i/j box, so the byte layout alone
    cannot disambiguate); callers reading a single file in isolation may
    pass it explicitly, or rely on the default `rank=0`.
    `load_reference`, which reads a full dump DIRECTORY, parses rank from
    each file's name (the `amps_dump_r{RRRRRR}_t{TTT}.bin` convention) and
    passes it through, so `RefDataset`'s rank-qualified pairing is always
    correct for real multi-rank dumps.
    """
    buf = Path(path).read_bytes()
    bo = "<" if len(buf) < 4 else _detect_byte_order(buf, path)
    c = _Cursor(buf, bo)
    records: list[Record] = []
    while not c.eof():
        magic = c.i4()
        if magic == MAGIC_MICRO:
            records.append(_read_micro(c, rank))
        elif magic == MAGIC_SED:
            records.append(_read_sed(c, rank))
        else:
            raise ValueError(f"{path}: bad magic {magic} at byte {c.pos - 4}")
    return records


_FNAME_RE = re.compile(r"amps_dump_r(\d+)_t(\d+)\.bin$")


def _parse_fname(path: Path) -> tuple[int, int]:
    """Extract (rank, thread) from an amps_dump_r{RRRRRR}_t{TTT}.bin name."""
    m = _FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"{path}: filename doesn't match amps_dump_r{{rank}}_t{{thread}}.bin")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# RefDataset: rank-aware pre/post pairing over a collection of records,
# regardless of whether they came from raw .bin dumps or a converted .npz.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RefDataset:
    """A collection of `MicroRecord`/`SedRecord` instances (from one or
    more dump files, or one converted `.npz`), with rank-qualified
    pre/post pairing. Build via `load_reference`, not directly."""

    micro: list[MicroRecord] = dataclasses.field(default_factory=list)
    sed: list[SedRecord] = dataclasses.field(default_factory=list)

    def micro_pairs(self) -> Iterator[tuple[MicroRecord, MicroRecord]]:
        """Yield (pre, post) `MicroRecord` pairs sharing the same
        `(rank, TIME_AMPS, i, j)` key, sorted by that key. A key with only
        one phase present is silently skipped (no partial pair to yield).
        Raises `ValueError` if two records ever collide on the identical
        `(rank, TIME_AMPS, i, j, phase)` key -- that would otherwise
        silently drop one of them (the scale_amps reader's `aggregate()`
        guards against the analogous flat-dict-key collision; this is the
        same safety property for the typed, list-based representation)."""
        by_key: dict[tuple[int, int, int, int], dict[int, MicroRecord]] = {}
        for rec in self.micro:
            key = (rec.rank, rec.TIME_AMPS, rec.i, rec.j)
            phases = by_key.setdefault(key, {})
            if rec.phase in phases:
                raise ValueError(
                    f"duplicate micro record for (rank={rec.rank}, t={rec.TIME_AMPS}, "
                    f"i={rec.i}, j={rec.j}, phase={rec.phase}): two source records collided "
                    "on the same rank-qualified key"
                )
            phases[rec.phase] = rec
        for key in sorted(by_key):
            phases = by_key[key]
            if MicroRecord.PHASE_PRE in phases and MicroRecord.PHASE_POST in phases:
                yield phases[MicroRecord.PHASE_PRE], phases[MicroRecord.PHASE_POST]

    def sed_pairs(self) -> Iterator[tuple[SedRecord, SedRecord]]:
        """Yield (pre, post) `SedRecord` pairs sharing the same
        `(rank, TIME_AMPS, i, j, isn)` key, sorted by that key. Same
        skip-if-incomplete / raise-if-duplicate behavior as `micro_pairs`."""
        by_key: dict[tuple[int, int, int, int, int], dict[int, SedRecord]] = {}
        for rec in self.sed:
            key = (rec.rank, rec.TIME_AMPS, rec.i, rec.j, rec.isn)
            phases = by_key.setdefault(key, {})
            if rec.phase in phases:
                raise ValueError(
                    f"duplicate sed record for (rank={rec.rank}, t={rec.TIME_AMPS}, "
                    f"i={rec.i}, j={rec.j}, isn={rec.isn}, phase={rec.phase}): two source "
                    "records collided on the same rank-qualified key"
                )
            phases[rec.phase] = rec
        for key in sorted(by_key):
            phases = by_key[key]
            if SedRecord.PHASE_PRE in phases and SedRecord.PHASE_POST in phases:
                yield phases[SedRecord.PHASE_PRE], phases[SedRecord.PHASE_POST]


def load_reference(npz_or_dir: str | Path) -> RefDataset:
    """Read reference data into a `RefDataset`, from EITHER a directory of
    raw `amps_dump_r*_t*.bin` files OR one converted `.npz` archive (as
    produced by scale_amps's `scripts/amps_dump_reader.py --output`)."""
    path = Path(npz_or_dir)
    if path.is_dir():
        return _load_reference_dir(path)
    if path.suffix == ".npz":
        return _load_reference_npz(path)
    raise ValueError(f"{path}: expected a directory of amps_dump_r*_t*.bin files or a .npz file")


def _load_reference_dir(dump_dir: Path) -> RefDataset:
    files = sorted(dump_dir.glob("amps_dump_r*_t*.bin"))
    if not files:
        raise ValueError(f"no amps_dump_r*_t*.bin files in {dump_dir}")
    dataset = RefDataset()
    for f in files:
        rank, _thread = _parse_fname(f)
        for rec in read_dump_file(f, rank=rank):
            if isinstance(rec, MicroRecord):
                dataset.micro.append(rec)
            else:
                dataset.sed.append(rec)
    return dataset


# Mirrors the scale_amps reader's `_key()` naming convention exactly (see
# that module's docstring): micro keys have no `isn` group, sed keys do.
_MICRO_KEY_RE = re.compile(r"^micro_r(\d+)_t(-?\d+)_i(-?\d+)_j(-?\d+)_(pre|post)_(.+)$")
_SED_KEY_RE = re.compile(r"^sed_r(\d+)_t(-?\d+)_i(-?\d+)_j(-?\d+)_s(\d+)_(pre|post)_(.+)$")


def _micro_record_from_fields(rank: int, fields: dict[str, np.ndarray]) -> MicroRecord:
    kwargs: dict[str, object] = {"rank": rank}
    for name in MICRO_HEADER:
        kwargs[name] = int(fields[name])
    kwargs["dt"] = float(fields["dt"])
    kwargs["kmicvm"] = fields["kmicvm"]
    for name in MICRO_R1:
        kwargs[name] = fields[name]
    for name in MICRO_R4:
        kwargs[name] = fields[name]
    if kwargs["phase"] == MicroRecord.PHASE_POST:
        for name, _ndims in MICRO_POST_EXTRA:
            kwargs[name] = fields[name]
    return MicroRecord(**kwargs)  # type: ignore[arg-type]


def _sed_record_from_fields(rank: int, fields: dict[str, np.ndarray]) -> SedRecord:
    kwargs: dict[str, object] = {"rank": rank}
    for name in SED_HEADER:
        kwargs["nprop" if name == "np" else name] = int(fields[name])
    kwargs["dt"] = float(fields["dt"])
    kwargs["k1b"] = fields["k1b"]
    kwargs["k2b"] = fields["k2b"]
    kwargs["qpv"] = fields["qpv"]
    for name in SED_R1:
        kwargs[name] = fields[name]
    kwargs["mmass"] = fields["mmass"]
    for name in SED_R1_TAIL:
        kwargs[name] = fields[name]
    kwargs["sflx"] = float(fields["sflx"])
    return SedRecord(**kwargs)  # type: ignore[arg-type]


def _load_reference_npz(npz_path: Path) -> RefDataset:
    """Reconstruct typed records from a flat, rank-qualified npz archive
    (scale_amps `aggregate()`'s output): every array is stored under a key
    `f"{base}_{field}"`, `base` one of `micro_r{rank}_t{t}_i{i}_j{j}_{pre|post}`
    / `sed_r{rank}_t{t}_i{i}_j{j}_s{isn}_{pre|post}` -- see `_MICRO_KEY_RE`/
    `_SED_KEY_RE`, which invert that exact convention."""
    dataset = RefDataset()
    micro_groups: dict[tuple[int, int, int, int, str], dict[str, np.ndarray]] = {}
    sed_groups: dict[tuple[int, int, int, int, int, str], dict[str, np.ndarray]] = {}

    with np.load(npz_path) as npz:
        for key in npz.files:
            m = _MICRO_KEY_RE.match(key)
            if m:
                rank_s, t_s, i_s, j_s, _phase_str, field = m.groups()
                micro_group_key = (int(rank_s), int(t_s), int(i_s), int(j_s), _phase_str)
                micro_groups.setdefault(micro_group_key, {})[field] = npz[key]
                continue
            s = _SED_KEY_RE.match(key)
            if s:
                rank_s, t_s, i_s, j_s, isn_s, phase_str, field = s.groups()
                sed_group_key = (int(rank_s), int(t_s), int(i_s), int(j_s), int(isn_s), phase_str)
                sed_groups.setdefault(sed_group_key, {})[field] = npz[key]
                continue
            raise ValueError(f"{npz_path}: key {key!r} doesn't match micro_*/sed_* naming")

        for (rank, _t, _i, _j, _phase_str), fields in micro_groups.items():
            dataset.micro.append(_micro_record_from_fields(rank, fields))
        for (rank, _t, _i, _j, _isn, _phase_str), fields in sed_groups.items():
            dataset.sed.append(_sed_record_from_fields(rank, fields))

    return dataset


# ---------------------------------------------------------------------------
# SetupRecord: one-shot `AMPS_DUMP_setup` dump (magic `MAGIC_SETUP`) --
# the Low-List collisional-breakup fragment tables (`bu_fd`/`bu_tmass`) +
# their four index scalars + the liquid bin boundaries, computed ONCE at
# AMPS setup (M2b Task 6 validation instrumentation, NOT part of the
# per-column micro/sed record stream above: one record, one dedicated file
# `amps_dump_setup.bin`, rank 0 only -- see `scale_atmos_phy_mp_amps.F90`'s
# own `AMPS_DUMP_setup` subroutine docstring for the write side).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SetupRecord:
    """One `AMPS_DUMP_setup` dump record: `cal_breakfragment`'s output
    for one liquid bin-grid/air-state configuration (cloudlab: `nbr=40`).
    `bu_tmass` has shape `(i1d_pair_max,)`, `bu_fd` has shape `(2,
    kk_max)`, `binbr` has shape `(nbr+1,)` -- ALL as the Fortran
    ALLOCATED them (a fixed, possibly-oversized capacity per
    `mod_amps_lib.F90`'s own `nbr==40`/`nbr==80` branch, NOT tightly
    sized to `i1d_pair_max`/`kk_max` -- compare against
    `core.lookup_tables.breakup_fragment_table_sizes(nbr, jmin_bk)`-sliced
    prefixes, not the raw arrays' own `.shape`)."""

    nbr: int
    jmin_bk: int
    imin_bk: int
    imax_bk: int
    jmax_bk: int
    binbr: np.ndarray
    bu_tmass: np.ndarray
    bu_fd: np.ndarray


def read_setup_dump(path: str | Path) -> SetupRecord:
    """Parse one `amps_dump_setup.bin` file (written by
    `AMPS_DUMP_setup`) into a `SetupRecord`. Single-record file (no
    magic-tagged stream loop, unlike `read_dump_file`); byte order is
    auto-detected off the leading magic, same convention as
    `_detect_byte_order`."""
    buf = Path(path).read_bytes()
    raw = int(np.frombuffer(buf, dtype="<i4", count=1, offset=0)[0])
    if raw == MAGIC_SETUP:
        bo = "<"
    else:
        swapped = int(np.frombuffer(buf, dtype=">i4", count=1, offset=0)[0])
        if swapped != MAGIC_SETUP:
            raise ValueError(
                f"{path}: bad magic {raw} at byte 0 (expected MAGIC_SETUP={MAGIC_SETUP})"
            )
        bo = ">"

    c = _Cursor(buf, bo)
    magic = c.i4()
    assert magic == MAGIC_SETUP
    version = int(c.i4())
    if version != 1:
        raise ValueError(f"unsupported setup record version {version}")
    nbr = int(c.i4())
    jmin_bk = int(c.i4())
    imin_bk = int(c.i4())
    imax_bk = int(c.i4())
    jmax_bk = int(c.i4())
    binbr = c.rn_arr(1)
    bu_tmass = c.rn_arr(1)
    bu_fd = c.rn_arr(2)
    return SetupRecord(
        nbr=nbr,
        jmin_bk=jmin_bk,
        imin_bk=imin_bk,
        imax_bk=imax_bk,
        jmax_bk=jmax_bk,
        binbr=binbr,
        bu_tmass=bu_tmass,
        bu_fd=bu_fd,
    )
