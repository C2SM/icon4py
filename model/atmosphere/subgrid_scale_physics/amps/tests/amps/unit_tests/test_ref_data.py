# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for driver/ref_data.py and driver/box.py (M1 Task 8).

pytest-converted round-trip suite covering the SAME scenarios as
scale_amps's `scripts/test_amps_dump_reader.py` (little+big endian, two
ranks, direct `dt`/`kmicvm`/`k1b`/`k2b` asserts), plus this module's own
additions: rank-aware pre/post pairing (`RefDataset.micro_pairs()`/
`.sed_pairs()`), the npz read path (`load_reference` on a converted
`.npz`, not just a raw-dump directory), and `driver/box.py`'s
`case_from_micro_record` field mapping.

The byte-builders below (`w_i0`/`w_i1`/`w_r0`/`w_rn`/`build_micro`/
`build_sed`) are COPIED from `scale_amps/scripts/test_amps_dump_reader.py`
(byte-identical to what the instrumented Fortran writer produces) -- keep
them in sync with that file if the on-disk record layout (`MICRO_*`/
`SED_*` field order/dims in `ref_data.py`) ever changes; scale_amps's own
copy stays authoritative for the cluster-side reader it tests.
"""

from __future__ import annotations

import dataclasses
import struct

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps
from icon4py.model.atmosphere.subgrid_scale_physics.amps.driver import box, ref_data
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import ThermoProp, ThermoState


# ---------------------------------------------------------------------------
# Byte-builders -- copied from scale_amps/scripts/test_amps_dump_reader.py;
# keep in sync with scale_amps (see module docstring).
# ---------------------------------------------------------------------------


def w_i0(b, v, bo="<"):
    return b + struct.pack(f"{bo}i", v)


def w_i1(b, a, bo="<"):
    a = np.asarray(a, dtype=f"{bo}i4")
    return b + struct.pack(f"{bo}i", a.size) + a.tobytes()


def w_r0(b, v, bo="<"):
    return b + struct.pack(f"{bo}d", v)


def w_rn(b, a, bo="<"):
    """Array with per-rank int32 size prefixes, Fortran order."""
    a = np.asarray(a, dtype=f"{bo}f8")
    for s in a.shape:
        b += struct.pack(f"{bo}i", s)
    return b + a.tobytes(order="F")


def build_micro(  # noqa: PLR0917 [too-many-positional-arguments]
    phase,
    nmic=3,
    npr=6,
    nbr=4,
    ncr=1,
    npi=18,
    nbi=2,
    nci=1,
    npa=5,
    nba=1,
    nca=4,
    mxnbin=4,
    bo="<",
):
    rng = np.random.default_rng(phase)
    b = b""
    for v in (
        ref_data.MAGIC_MICRO,
        1,
        phase,
        42,
        3,
        4,
        1,
        nmic,
        npr,
        nbr,
        ncr,
        npi,
        nbi,
        nci,
        npa,
        nba,
        nca,
        mxnbin,
        1,
        12345,
        0,
        7,
        99,
    ):
        b = w_i0(b, v, bo)
    b = w_r0(b, 1.0, bo)
    kmicvm = np.arange(2, 2 + nmic)
    b = w_i1(b, kmicvm, bo)
    # dt/kmicvm are tracked here (not just consumed silently) so the
    # round-trip loop below asserts them directly, same as every other field.
    fields = {"dt": 1.0, "kmicvm": kmicvm}
    for name in (
        "qcvm",
        "v3v",
        "qvvm",
        "moist_denvm",
        "ptotvm",
        "tvm",
        "wbvm",
        "trpv_thil",
        "trpv_qtp",
    ):
        fields[name] = rng.uniform(size=nmic)
        b = w_rn(b, fields[name], bo)
    fields["qrpvm"] = rng.uniform(size=(npr, nbr, ncr, nmic))
    fields["qipvm"] = rng.uniform(size=(npi, nbi, nci, nmic))
    fields["qapvm"] = rng.uniform(size=(npa, nba, nca, nmic))
    for name in ("qrpvm", "qipvm", "qapvm"):
        b = w_rn(b, fields[name], bo)
    if phase == 2:
        fields["dmtendlm"] = rng.uniform(size=(10, 2, nmic))
        fields["dcontendlm"] = rng.uniform(size=(10, 2, nmic))
        fields["dbintendlm"] = rng.uniform(size=(7, 2, mxnbin, nmic))
        for name in ("dmtendlm", "dcontendlm", "dbintendlm"):
            b = w_rn(b, fields[name], bo)
    return b, fields


def build_sed(phase, isn=0, np_=6, nb=4, nc=1, nzh=8, bo="<"):  # noqa: PLR0917 [too-many-positional-arguments]
    rng = np.random.default_rng(100 + phase)
    b = b""
    for v in (ref_data.MAGIC_SED, 1, phase, 42, 3, 4, 1, isn, 1, np_, nb, nc, 2, 5, 2, 6):
        b = w_i0(b, v, bo)
    b = w_r0(b, 1.0, bo)
    # Non-constant so a wrong reshape order ("C" instead of "F") would be
    # caught when nc > 1, not just a wrong shape.
    k1b_flat = np.arange(10, 10 + nb * nc)
    k2b_flat = np.arange(50, 50 + nb * nc)
    b = w_i1(b, k1b_flat, bo)
    b = w_i1(b, k2b_flat, bo)
    fields = {
        "dt": 1.0,
        "k1b": k1b_flat.reshape((nb, nc), order="F"),
        "k2b": k2b_flat.reshape((nb, nc), order="F"),
    }
    fields["qpv"] = rng.uniform(size=(np_, nb, nc, nzh))
    b = w_rn(b, fields["qpv"], bo)
    r1names = (
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
    for name in r1names:
        fields[name] = rng.uniform(size=nzh)
        b = w_rn(b, fields[name], bo)
    fields["mmass"] = rng.uniform(size=(nb, nc, nzh))
    b = w_rn(b, fields["mmass"], bo)
    for name in ("den_t", "momz_t", "rhou_t", "rhov_t", "rhoe_t"):
        fields[name] = rng.uniform(size=nzh)
        b = w_rn(b, fields[name], bo)
    b = w_r0(b, 3.5, bo)
    fields["sflx"] = 3.5
    return b, fields


# ---------------------------------------------------------------------------
# read_dump_file: round trip, byte-order auto-detect.
# ---------------------------------------------------------------------------


class TestReadDumpFile:
    def test_round_trip(self, tmp_path):
        """Mixed micro+sed records, single rank/thread; direct dt/kmicvm/
        k1b/k2b asserts (previously only indirectly exercised, if at all)."""
        blob1, f1 = build_micro(1)
        blob2, f2 = build_micro(2)
        blob3, f3 = build_sed(3, nc=2)  # nc>1 exercises the (nb, nc) F-order reshape
        blob4, f4 = build_sed(4, isn=1, nc=2)
        p = tmp_path / "amps_dump_r000000_t001.bin"
        p.write_bytes(blob1 + blob3 + blob4 + blob2)
        recs = ref_data.read_dump_file(p)

        assert len(recs) == 4
        assert [type(r).__name__ for r in recs] == [
            "MicroRecord",
            "SedRecord",
            "SedRecord",
            "MicroRecord",
        ]
        assert recs[0].phase == 1
        assert recs[3].phase == 2
        assert recs[0].TIME_AMPS == 42
        assert recs[0].i == 3
        assert recs[0].j == 4
        assert recs[0].rank == 0  # default, since no rank= was passed

        for expected, rec in ((f1, recs[0]), (f2, recs[3]), (f3, recs[1]), (f4, recs[2])):
            for name, val in expected.items():
                got = getattr(rec, name)
                assert np.allclose(got, val), f"{type(rec).__name__} field {name} mismatch"

        assert recs[1].isn == 0
        assert recs[2].isn == 1

        # Direct asserts (dt/kmicvm/k1b/k2b).
        assert recs[0].dt == 1.0
        assert np.array_equal(recs[0].kmicvm, np.arange(2, 2 + 3))
        assert recs[1].k1b.shape == (4, 2)
        assert recs[1].k2b.shape == (4, 2)
        assert np.array_equal(recs[1].k1b, f3["k1b"])
        assert np.array_equal(recs[1].k2b, f3["k2b"])

    def test_byteswap(self, tmp_path):
        """A big-endian file (as produced by SCALE cluster builds compiled
        with -fconvert=big-endian / -convert big_endian) must parse
        identically to the little-endian one, via auto-detection off the
        record magic."""
        _blob_le, fields_le = build_micro(1, bo="<")
        blob_be, fields_be = build_micro(1, bo=">")
        assert all(np.allclose(fields_le[k], fields_be[k]) for k in fields_le), (
            "test bug: le/be field builders diverged"
        )
        p = tmp_path / "amps_dump_r000002_t001.bin"
        p.write_bytes(blob_be)
        recs = ref_data.read_dump_file(p, rank=2)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.rank == 2
        for name, val in fields_be.items():
            assert np.allclose(getattr(rec, name), val), f"byteswapped field {name} mismatch"

    def test_bad_magic_raises_value_error(self, tmp_path):
        p = tmp_path / "amps_dump_r000000_t001.bin"
        p.write_bytes(struct.pack("<i", 999))
        with pytest.raises(ValueError, match="bad magic"):
            ref_data.read_dump_file(p)

    def test_bad_version_raises_value_error(self, tmp_path):
        """Version check must be `raise ValueError`, not `assert` -- must
        survive `python -O` (carry-forward #2)."""
        b = w_i0(b"", ref_data.MAGIC_MICRO)
        b = w_i0(b, 2)  # unsupported version
        p = tmp_path / "amps_dump_r000000_t001.bin"
        p.write_bytes(b)
        with pytest.raises(ValueError, match="unsupported micro record version"):
            ref_data.read_dump_file(p)


# ---------------------------------------------------------------------------
# load_reference: raw-dump directory path.
# ---------------------------------------------------------------------------


class TestLoadReferenceDir:
    def test_two_ranks(self, tmp_path):
        """Two ranks dumping the SAME local (i, j) box (the normal case)
        must both survive as distinct records tagged with their own rank
        -- not collide or get merged."""
        blob, fields = build_micro(1)
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(blob)
        (tmp_path / "amps_dump_r000001_t001.bin").write_bytes(blob)

        dataset = ref_data.load_reference(tmp_path)

        assert len(dataset.micro) == 2
        assert len(dataset.sed) == 0
        assert sorted(r.rank for r in dataset.micro) == [0, 1]
        for rec in dataset.micro:
            assert rec.dt == fields["dt"] == 1.0
            assert np.array_equal(rec.kmicvm, fields["kmicvm"])

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no amps_dump_r"):
            ref_data.load_reference(tmp_path)

    def test_unrecognized_path_raises(self, tmp_path):
        bad = tmp_path / "not_a_dir_or_npz.txt"
        bad.write_text("x")
        with pytest.raises(ValueError, match="expected a directory"):
            ref_data.load_reference(bad)


# ---------------------------------------------------------------------------
# RefDataset pairing: (rank, t, i, j[, isn]) matching.
# ---------------------------------------------------------------------------


class TestPairing:
    def test_micro_pairs_matches_by_rank_t_i_j(self, tmp_path):
        pre, _ = build_micro(1)
        post, _ = build_micro(2)
        # Two ranks, each with a pre/post pair for the SAME (t, i, j).
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(pre + post)
        (tmp_path / "amps_dump_r000001_t001.bin").write_bytes(pre + post)

        dataset = ref_data.load_reference(tmp_path)
        pairs = sorted(dataset.micro_pairs(), key=lambda pair: pair[0].rank)

        assert len(pairs) == 2
        for rank, (pre_rec, post_rec) in enumerate(pairs):
            assert pre_rec.rank == post_rec.rank == rank
            assert pre_rec.phase == ref_data.MicroRecord.PHASE_PRE
            assert post_rec.phase == ref_data.MicroRecord.PHASE_POST
            assert (pre_rec.TIME_AMPS, pre_rec.i, pre_rec.j) == (42, 3, 4)

    def test_sed_pairs_matches_by_rank_t_i_j_isn(self, tmp_path):
        pre, _ = build_sed(3, isn=1)
        post, _ = build_sed(4, isn=1)
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(pre + post)

        dataset = ref_data.load_reference(tmp_path)
        pairs = list(dataset.sed_pairs())

        assert len(pairs) == 1
        pre_rec, post_rec = pairs[0]
        assert pre_rec.phase == ref_data.SedRecord.PHASE_PRE
        assert post_rec.phase == ref_data.SedRecord.PHASE_POST
        assert pre_rec.isn == post_rec.isn == 1
        assert (pre_rec.TIME_AMPS, pre_rec.i, pre_rec.j) == (42, 3, 4)

    def test_incomplete_pair_is_skipped(self, tmp_path):
        pre, _ = build_micro(1)
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(pre)
        dataset = ref_data.load_reference(tmp_path)
        assert len(dataset.micro) == 1
        assert list(dataset.micro_pairs()) == []

    def test_duplicate_key_guard(self, tmp_path):
        """Two files that resolve to the same rank and produce the same
        record key (two different thread-files under rank 0, same
        (TIME_AMPS, i, j)) must raise when paired, not silently drop one
        -- carries forward the scale_amps reader's `aggregate()` duplicate-
        key guard, adapted for the list-based `RefDataset` (see
        `micro_pairs`'s own docstring for why load-time itself can't
        collide the way a flat dict could)."""
        blob, _ = build_micro(1)
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(blob)
        (tmp_path / "amps_dump_r000000_t002.bin").write_bytes(blob)

        dataset = ref_data.load_reference(tmp_path)
        assert len(dataset.micro) == 2  # both survive as distinct list entries...

        with pytest.raises(ValueError, match="duplicate micro record"):
            list(dataset.micro_pairs())  # ...but pairing must catch the collision


# ---------------------------------------------------------------------------
# load_reference: converted .npz path -- must reconstruct the SAME records
# as reading the raw dump directory directly.
# ---------------------------------------------------------------------------


def _flatten_micro_record(rec: ref_data.MicroRecord, rank: int) -> dict:
    """Rebuild the SAME rank-qualified flat-key convention scale_amps's
    `aggregate()` produces, from an already-parsed typed record -- used
    only to synthesize an npz fixture for `TestLoadReferenceNpz` (mirrors,
    does not import, scale_amps's own `aggregate()`/`_key()`)."""
    phase_str = "pre" if rec.phase == ref_data.MicroRecord.PHASE_PRE else "post"
    base = f"micro_r{rank}_t{rec.TIME_AMPS}_i{rec.i}_j{rec.j}_{phase_str}"
    flat = {f"{base}_{name}": getattr(rec, name) for name in ref_data.MICRO_HEADER}
    flat[f"{base}_dt"] = rec.dt
    flat[f"{base}_kmicvm"] = rec.kmicvm
    for name in ref_data.MICRO_R1 + ref_data.MICRO_R4:
        flat[f"{base}_{name}"] = getattr(rec, name)
    if rec.phase == ref_data.MicroRecord.PHASE_POST:
        for name, _ndims in ref_data.MICRO_POST_EXTRA:
            flat[f"{base}_{name}"] = getattr(rec, name)
    return flat


def _flatten_sed_record(rec: ref_data.SedRecord, rank: int) -> dict:
    phase_str = "pre" if rec.phase == ref_data.SedRecord.PHASE_PRE else "post"
    base = f"sed_r{rank}_t{rec.TIME_AMPS}_i{rec.i}_j{rec.j}_s{rec.isn}_{phase_str}"
    flat = {}
    for name in ref_data.SED_HEADER:
        attr = "nprop" if name == "np" else name
        flat[f"{base}_{name}"] = getattr(rec, attr)
    flat[f"{base}_dt"] = rec.dt
    flat[f"{base}_k1b"] = rec.k1b
    flat[f"{base}_k2b"] = rec.k2b
    flat[f"{base}_qpv"] = rec.qpv
    for name in ref_data.SED_R1:
        flat[f"{base}_{name}"] = getattr(rec, name)
    flat[f"{base}_mmass"] = rec.mmass
    for name in ref_data.SED_R1_TAIL:
        flat[f"{base}_{name}"] = getattr(rec, name)
    flat[f"{base}_sflx"] = rec.sflx
    return flat


class TestLoadReferenceNpz:
    def test_npz_matches_dir(self, tmp_path):
        micro_pre, _ = build_micro(1)
        micro_post, _ = build_micro(2)
        sed_pre, _ = build_sed(3, isn=0, nc=2)
        sed_post, _ = build_sed(4, isn=0, nc=2)
        (tmp_path / "amps_dump_r000000_t001.bin").write_bytes(
            micro_pre + micro_post + sed_pre + sed_post
        )

        dir_dataset = ref_data.load_reference(tmp_path)
        assert len(dir_dataset.micro) == 2
        assert len(dir_dataset.sed) == 2

        flat: dict = {}
        for rec in dir_dataset.micro:
            flat.update(_flatten_micro_record(rec, rec.rank))
        for rec in dir_dataset.sed:
            flat.update(_flatten_sed_record(rec, rec.rank))
        npz_path = tmp_path / "amps_ref.npz"
        np.savez_compressed(npz_path, **flat)

        npz_dataset = ref_data.load_reference(npz_path)
        assert len(npz_dataset.micro) == 2
        assert len(npz_dataset.sed) == 2

        dir_micro = {(r.rank, r.TIME_AMPS, r.i, r.j, r.phase): r for r in dir_dataset.micro}
        npz_micro = {(r.rank, r.TIME_AMPS, r.i, r.j, r.phase): r for r in npz_dataset.micro}
        assert set(dir_micro) == set(npz_micro)
        for key, dir_rec in dir_micro.items():
            npz_rec = npz_micro[key]
            for name in ref_data.MICRO_HEADER:
                assert getattr(dir_rec, name) == getattr(npz_rec, name)
            assert dir_rec.dt == npz_rec.dt
            assert np.array_equal(dir_rec.kmicvm, npz_rec.kmicvm)
            for name in ref_data.MICRO_R1 + ref_data.MICRO_R4:
                assert np.allclose(getattr(dir_rec, name), getattr(npz_rec, name))

        dir_sed = {(r.rank, r.TIME_AMPS, r.i, r.j, r.isn, r.phase): r for r in dir_dataset.sed}
        npz_sed = {(r.rank, r.TIME_AMPS, r.i, r.j, r.isn, r.phase): r for r in npz_dataset.sed}
        assert set(dir_sed) == set(npz_sed)
        for key, dir_rec in dir_sed.items():
            npz_rec = npz_sed[key]
            assert dir_rec.nprop == npz_rec.nprop
            assert dir_rec.dt == npz_rec.dt
            assert np.array_equal(dir_rec.k1b, npz_rec.k1b)
            assert np.array_equal(dir_rec.k2b, npz_rec.k2b)
            for name in ref_data.SED_R1 + ref_data.SED_R1_TAIL:
                assert np.allclose(getattr(dir_rec, name), getattr(npz_rec, name))

    def test_unrecognized_npz_key_raises(self, tmp_path):
        npz_path = tmp_path / "bad.npz"
        np.savez_compressed(npz_path, not_a_record_key=np.array([1.0]))
        with pytest.raises(ValueError, match="doesn't match micro_\\*/sed_\\* naming"):
            ref_data.load_reference(npz_path)


# ---------------------------------------------------------------------------
# box.case_from_micro_record: field mapping (pre-record -> BoxCase).
# ---------------------------------------------------------------------------


def _make_pre_micro_record(nmic: int = 4, *, rng_seed: int = 7) -> ref_data.MicroRecord:
    """A synthetic phase=1 ("pre") MicroRecord with npr/npi/npa matching
    the real LiquidPPV/IcePPV/AerosolPPV lengths (unlike `build_micro`'s
    defaults, which use arbitrary sizes purely for byte-layout testing) --
    the shape `case_from_micro_record` actually requires."""
    rng = np.random.default_rng(rng_seed)
    npr, npi, npa = len(index_maps.LiquidPPV), len(index_maps.IcePPV), len(index_maps.AerosolPPV)
    nbr, nbi, nba = 3, 2, 2
    ptotvm = rng.uniform(8.0e4, 1.0e5, size=nmic)
    tvm = rng.uniform(250.0, 290.0, size=nmic)
    qvvm = rng.uniform(1.0e-4, 5.0e-3, size=nmic)
    return ref_data.MicroRecord(
        rank=0,
        phase=ref_data.MicroRecord.PHASE_PRE,
        TIME_AMPS=42,
        i=3,
        j=4,
        isect=1,
        nmic=nmic,
        npr=npr,
        nbr=nbr,
        ncr=1,
        npi=npi,
        nbi=nbi,
        nci=1,
        npa=npa,
        nba=nba,
        nca=1,
        mxnbin=4,
        istrt=1,
        jseed=12345,
        ifrst=0,
        isect_seed=7,
        nextn=99,
        dt=2.5,
        kmicvm=np.arange(1, 1 + nmic, dtype=np.int32),
        qcvm=rng.uniform(size=nmic),
        v3v=rng.uniform(size=nmic),
        qvvm=qvvm,
        moist_denvm=rng.uniform(0.5, 1.5, size=nmic),
        ptotvm=ptotvm,
        tvm=tvm,
        wbvm=rng.uniform(-1.0, 1.0, size=nmic),
        trpv_thil=rng.uniform(size=nmic),
        trpv_qtp=rng.uniform(size=nmic),
        qrpvm=rng.uniform(1.0e-7, 1.0e-5, size=(npr, nbr, 1, nmic)),
        qipvm=rng.uniform(1.0e-7, 1.0e-5, size=(npi, nbi, 1, nmic)),
        qapvm=rng.uniform(1.0e-7, 1.0e-5, size=(npa, nba, 1, nmic)),
    )


def _thermo_prop(thermo: ThermoState, prop: ThermoProp) -> np.ndarray:
    idx = ThermoState.PROPS.index(prop)
    return thermo.values[idx, 0, 0, :]


class TestCaseFromMicroRecord:
    def test_field_mapping(self):
        rec = _make_pre_micro_record(nmic=4)
        case = box.case_from_micro_record(rec, n_steps=3)

        assert case.dt == rec.dt
        assert case.n_steps == 3
        assert case.config == AmpsConfig.cloudlab()
        assert np.array_equal(case.liquid.values, rec.qrpvm)
        assert np.array_equal(case.ice.values, rec.qipvm)
        assert np.array_equal(case.aerosol.values, rec.qapvm)

        thermo = case.thermo
        assert np.array_equal(_thermo_prop(thermo, ThermoProp.ptotv), rec.ptotvm)
        assert np.array_equal(_thermo_prop(thermo, ThermoProp.tv), rec.tvm)
        assert np.array_equal(_thermo_prop(thermo, ThermoProp.qvv), rec.qvvm)
        assert np.array_equal(_thermo_prop(thermo, ThermoProp.moist_denv), rec.moist_denvm)
        assert np.array_equal(_thermo_prop(thermo, ThermoProp.wbv), rec.wbvm)
        assert np.allclose(_thermo_prop(thermo, ThermoProp.pbv), 0.0)
        # FACT-GAP: momv is never captured by AMPS_DUMP_micro (see box.py's
        # module docstring) -- defaulted to zero, not silently wrong.
        assert np.allclose(_thermo_prop(thermo, ThermoProp.momv), 0.0)

        # thv/piv/thetav via the Exner relation (Z_LOOP_01 lines 1644/1647/
        # 1661), independently recomputed here (not just re-reading box.py's
        # own constants) so a wrong exponent/formula would be caught.
        rdry, cpdry, pre00 = 287.04, 1004.64, 1.0e5
        expected_thv = rec.tvm * (pre00 / rec.ptotvm) ** (rdry / cpdry)
        expected_piv = rec.tvm / expected_thv * cpdry
        expected_thetav = expected_thv * (1.0 + 0.61 * rec.qvvm)
        assert np.allclose(_thermo_prop(thermo, ThermoProp.thv), expected_thv)
        assert np.allclose(_thermo_prop(thermo, ThermoProp.piv), expected_piv)
        assert np.allclose(_thermo_prop(thermo, ThermoProp.thetav), expected_thetav)

    def test_default_config_is_cloudlab(self):
        rec = _make_pre_micro_record()
        case = box.case_from_micro_record(rec)
        assert case.config == AmpsConfig.cloudlab()

    def test_explicit_config_is_used(self):
        rec = _make_pre_micro_record()
        custom = AmpsConfig.cloudlab_seeding()
        case = box.case_from_micro_record(rec, config=custom)
        assert case.config == custom

    def test_default_n_steps_is_one(self):
        rec = _make_pre_micro_record()
        case = box.case_from_micro_record(rec)
        assert case.n_steps == 1

    def test_rejects_post_phase(self):
        rec = dataclasses.replace(_make_pre_micro_record(), phase=ref_data.MicroRecord.PHASE_POST)
        with pytest.raises(ValueError, match="phase"):
            box.case_from_micro_record(rec)

    def test_rejects_npr_mismatch(self):
        rec = dataclasses.replace(_make_pre_micro_record(), npr=999)
        with pytest.raises(ValueError, match="npr"):
            box.case_from_micro_record(rec)

    def test_rejects_npi_mismatch(self):
        rec = dataclasses.replace(_make_pre_micro_record(), npi=999)
        with pytest.raises(ValueError, match="npi"):
            box.case_from_micro_record(rec)

    def test_rejects_npa_mismatch(self):
        rec = dataclasses.replace(_make_pre_micro_record(), npa=999)
        with pytest.raises(ValueError, match="npa"):
            box.case_from_micro_record(rec)

    def test_rejects_ncr_not_one(self):
        """A dump record with ncat != 1 (e.g. ncr=2) must fail at this
        data-entry seam, not silently misinterpret qrpvm's category axis
        downstream (state.py's `_BinnedState`/F4 both pin ncat=1)."""
        rec = dataclasses.replace(_make_pre_micro_record(), ncr=2)
        with pytest.raises(ValueError, match="ncr"):
            box.case_from_micro_record(rec)


# ---------------------------------------------------------------------------
# BoxCase validation + run_box skeleton behavior.
# ---------------------------------------------------------------------------


class TestBoxCaseValidation:
    def test_mismatched_npoints_raises(self):
        case = box.case_from_micro_record(_make_pre_micro_record(nmic=4))
        bad_liquid = dataclasses.replace(case.liquid, values=case.liquid.values[:, :, :, :2])
        with pytest.raises(ValueError, match="npoints"):
            dataclasses.replace(case, liquid=bad_liquid)

    def test_nonpositive_dt_raises(self):
        case = box.case_from_micro_record(_make_pre_micro_record())
        with pytest.raises(ValueError, match="dt"):
            dataclasses.replace(case, dt=0.0)

    def test_nonpositive_n_steps_raises(self):
        case = box.case_from_micro_record(_make_pre_micro_record())
        with pytest.raises(ValueError, match="n_steps"):
            dataclasses.replace(case, n_steps=0)


def test_run_box_not_implemented():
    case = box.case_from_micro_record(_make_pre_micro_record())
    with pytest.raises(NotImplementedError):
        box.run_box(case)
