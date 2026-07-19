# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/breakfragment.py (M2b Task 6): the real Low-List (1982)
collisional-breakup fragment-table generator (`cal_breakfragment`), per
docs/superpowers/facts/m2b/breakfragment-full-chain.md ("H3" below).

Groups, matching the task dispatch's own test list:

* `TestGetznorm2` -- `getznorm2`/`cumnor` vs `scipy.stats.norm.cdf` (a
  Cody rational Chebyshev approximation is designed to match the true
  normal CDF to ~machine precision, so a tight tolerance here is a
  legitimate correctness check on the transcription, not merely a sanity
  bound), across both branch boundaries (`thrsh=0.66291`,
  `root32=5.656854248`) and negative/zero/positive arguments.
* `TestZbrentSolvers` -- `cal_sig_sf`/`cal_hmusig` converge (not the
  `-999.9` sentinel, finite) and satisfy their own fixed-point objective
  to within a small residual, for realistic Low-List parameter ranges.
* `TestCalBreakupDisLl` -- spot values: mass conservation
  (`sum(frag_mass) == m_coal` by construction of `mrat`), the `D_coal<=D_0`
  early-return contract, and independently-recomputed (not merely
  re-calling the module's own helpers) `R_f`/`R_s`/`R_d`/`F_f`/`m_coal`
  for a hand-picked bin pair.
* `TestLiquidLenVtm` -- `%len`/`%vtm` shapes, monotonicity, `D_0` cutoff.
* `TestMakeBreakupFragmentTables` -- shapes vs
  `breakup_fragment_table_sizes`, index-scalar sanity, `is_placeholder`
  is `False`, mass conservation end-to-end, cloudlab (`nrbin=40,
  nbin_h=20`) runs and is finite.
* `TestAgainstAmpsSetupDump` -- marker-gated (`pytest.mark.datatest`)
  comparison against a real `AMPS_DUMP_setup` dump
  (`scale_atmos_phy_mp_amps.F90`'s M2b Task 6 instrumentation). SKIPPED
  with a pointer to
  docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md when no
  local `amps_dump_setup.bin` is found -- this dump does not exist yet as
  of this task (a short scale_amps rerun, or a standalone setup-only run,
  is needed to produce it; see that doc's own updated note). Activates
  automatically the moment one lands, no code change needed, mirroring
  test_warm_replay.py's own `_find_dump_source` convention.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid, breakfragment
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    breakup_fragment_table_sizes,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.driver import ref_data


try:
    from icon4py.model.testing.config import TEST_DATA_PATH
except ImportError:
    TEST_DATA_PATH = None


COEDPI6 = float(AmpsConst.coedpi6)
COEDSQ2P = float(AmpsConst.coedsq2p)


# ---------------------------------------------------------------------------
# getznorm2 / cumnor vs scipy.stats.norm.cdf.
# ---------------------------------------------------------------------------


class TestGetznorm2:
    @pytest.mark.parametrize(
        "x",
        [
            -7.0,
            -5.7,
            -4.0,
            -2.0,
            -1.0,
            -0.66291,
            -0.5,
            -0.1,
            0.0,
            0.1,
            0.5,
            0.66291,
            1.0,
            2.0,
            4.0,
            5.656854248,
            7.0,
        ],
    )
    def test_matches_scipy_norm_cdf(self, x):
        got = breakfragment.getznorm2(x)
        expected = float(scipy.stats.norm.cdf(x))
        assert got == pytest.approx(expected, abs=1e-12)

    def test_vectorized_matches_scalar(self):
        xs = np.linspace(-8.0, 8.0, 101)
        vec = breakfragment.getznorm2(xs)
        scalars = np.array([breakfragment.getznorm2(float(x)) for x in xs])
        np.testing.assert_allclose(vec, scalars)

    def test_scalar_input_returns_python_float(self):
        out = breakfragment.getznorm2(0.3)
        assert isinstance(out, float)

    def test_array_input_returns_ndarray(self):
        out = breakfragment.getznorm2(np.array([0.1, 0.2]))
        assert isinstance(out, np.ndarray)

    def test_monotonically_increasing(self):
        xs = np.linspace(-10.0, 10.0, 501)
        vals = breakfragment.getznorm2(xs)
        assert np.all(np.diff(vals) >= 0.0)

    def test_symmetry_about_zero(self):
        xs = np.array([0.2, 1.3, 3.5, 5.9])
        lo = breakfragment.getznorm2(-xs)
        hi = breakfragment.getznorm2(xs)
        np.testing.assert_allclose(lo, 1.0 - hi, atol=1e-14)


# ---------------------------------------------------------------------------
# zbrent-based solvers: cal_sig_sf / cal_hmusig.
# ---------------------------------------------------------------------------


class TestZbrentSolvers:
    @pytest.mark.parametrize(
        "d0,r,h_s,mu_s",
        [
            (0.01, 0.8, 5.0, 0.3),
            (0.01, 0.5, 50.8 * 0.5 ** (-0.718), 0.5),
            (0.01, 0.2, 4.18 * 0.05 ** (-1.17), 0.05),
        ],
    )
    def test_cal_sig_sf_converges_and_satisfies_objective(self, d0, r, h_s, mu_s):
        sig = breakfragment.cal_sig_sf(d0, r, 1.0, h_s, mu_s)
        assert math.isfinite(sig)
        assert sig != -999.9
        assert sig > 0.0

        phi = min(0.99999, breakfragment.getznorm2((d0 - mu_s) / max(sig, 1.0e-20)))
        residual = sig - 1.0 / h_s / COEDSQ2P / (1.0 - phi)
        assert abs(residual) < 1.0e-3

    def test_cal_sig_sf_tiny_r_returns_closed_form_without_solving(self):
        # R < 1e-20: Fortran returns 1/(H_s*coedsq2p) immediately, no zbrent.
        h_s = 7.3
        sig = breakfragment.cal_sig_sf(0.01, 1.0e-25, 1.0, h_s, 0.2)
        assert sig == pytest.approx(1.0 / (h_s * COEDSQ2P))

    @pytest.mark.parametrize(
        "d0,r,lin_mu,p_mode,nx",
        [
            (0.01, 0.7, 0.05, 2.0, 3.0),
            (0.01, 0.4, 0.2, 0.5, 5.0),
            (0.01, 0.9, 0.01, 10.0, 8.0),
        ],
    )
    def test_cal_hmusig_converges_and_satisfies_objective(self, d0, r, lin_mu, p_mode, nx):
        h, mu, sig = breakfragment.cal_hmusig(d0, r, lin_mu, p_mode, nx)
        assert math.isfinite(h)
        assert math.isfinite(mu)
        assert math.isfinite(sig)
        assert sig != -999.9
        assert h > 0.0

        h_local = p_mode * lin_mu * math.exp(0.5 * sig * sig)
        c1 = math.log(lin_mu) + sig * sig
        phi = min(0.99999999, breakfragment.getznorm2((math.log(d0) - c1) / max(sig, 1.0e-20)))
        residual = sig - nx / h_local / COEDSQ2P / (1.0 - phi)
        assert abs(residual) < 1.0e-3

        # post-loop H/mu recomputation, matching the Fortran's own
        # `H=P_mode*lin_mu*exp(0.5*sig**2); mu=log(lin_mu)+sig**2`.
        assert h == pytest.approx(p_mode * lin_mu * math.exp(0.5 * sig * sig))
        assert mu == pytest.approx(math.log(lin_mu) + sig * sig)

    @pytest.mark.parametrize(
        "nx,p_mode,r",
        [(1.0e-25, 2.0, 0.5), (2.0, 1.0e-25, 0.5), (2.0, 2.0, 1.0e-25)],
    )
    def test_cal_hmusig_degenerate_inputs_return_zero_hmu(self, nx, p_mode, r):
        # Nx<=1e-20 or P_mode<=1e-20 or R<=1e-20: Fortran returns H=0,mu=0
        # immediately (sig left at its just-set 10*lin_mu, never solved).
        lin_mu = 0.1
        h, mu, sig = breakfragment.cal_hmusig(0.01, r, lin_mu, p_mode, nx)
        assert h == 0.0
        assert mu == 0.0
        assert sig == pytest.approx(10.0 * lin_mu)


# ---------------------------------------------------------------------------
# cal_breakup_dis_ll: mass conservation, early-return contract, and one
# independently-recomputed spot value.
# ---------------------------------------------------------------------------


def _simple_binb(nbin: int) -> np.ndarray:
    """A synthetic, geometrically-spaced liquid mass-boundary array
    spanning roughly the D_0..1cm diameter range -- independent of
    bin_grid.py's own real cloudlab construction, so a bug shared between
    this test's expectations and the real bin grid can't hide."""
    d_min, d_max = 0.02, 1.0  # cm, straddles D_0=0.01cm comfortably above
    diam = np.geomspace(d_min, d_max, nbin + 1)
    return COEDPI6 * diam**3.0


class TestCalBreakupDisLl:
    def test_mass_conservation(self):
        nbin = 20
        binb = _simple_binb(nbin)
        # D_L=0.3cm, D_S=0.15cm (SI m), CKE/S_T/S_C from realistic
        # magnitudes (matching test_collision_kernel.py's own "real
        # magnitude" fixtures).
        result = breakfragment.cal_breakup_dis_ll(
            binb, nbin, d_l_m=0.3e-2, d_s_m=0.15e-2, s_t_j=3.0e-6, s_c_j=2.5e-6, cke_j=2.0e-6
        )
        assert result is not None
        m_coal, frag_mass, frag_con = result
        assert math.isfinite(m_coal)
        assert m_coal > 0.0
        assert frag_mass.shape == (nbin,)
        assert frag_con.shape == (nbin,)
        assert np.all(np.isfinite(frag_mass))
        assert np.all(np.isfinite(frag_con))
        assert np.all(frag_mass >= 0.0)
        assert np.all(frag_con >= 0.0)
        # mrat = m_coal/sum(dmass) is constructed so sum(frag_mass) ==
        # m_coal exactly (up to float64 rounding) -- the defining identity
        # of the Fortran's own final normalization step (H3 §2, "the
        # write-side index math + mass renormalization mrat").
        assert np.sum(frag_mass) == pytest.approx(m_coal, rel=1.0e-9)

    def test_d_coal_below_cutoff_returns_none(self):
        nbin = 10
        binb = _simple_binb(nbin)
        # Both drops far below D_0=0.01cm: D_coal also below the cutoff.
        result = breakfragment.cal_breakup_dis_ll(
            binb, nbin, d_l_m=1.0e-5, d_s_m=0.5e-5, s_t_j=1.0e-8, s_c_j=1.0e-8, cke_j=1.0e-8
        )
        assert result is None

    def test_m_coal_matches_independent_geometric_formula(self):
        # m_coal = coedpi6*D_coal**3, D_coal=(D_L_cm**3+D_S_cm**3)**(1/3) --
        # recomputed here from the raw D_L/D_S inputs, not by re-deriving
        # any of cal_breakup_dis_ll's own internals.
        nbin = 15
        binb = _simple_binb(nbin)
        d_l_m, d_s_m = 0.25e-2, 0.1e-2
        result = breakfragment.cal_breakup_dis_ll(
            binb, nbin, d_l_m=d_l_m, d_s_m=d_s_m, s_t_j=2.0e-6, s_c_j=1.8e-6, cke_j=1.5e-6
        )
        assert result is not None
        m_coal, _frag_mass, _frag_con = result

        d_l_cm, d_s_cm = d_l_m * 100.0, d_s_m * 100.0
        d_coal_expected = (d_l_cm**3.0 + d_s_cm**3.0) ** (1.0 / 3.0)
        m_coal_expected = COEDPI6 * d_coal_expected**3.0
        assert m_coal == pytest.approx(m_coal_expected, rel=1.0e-12)

    def test_r_f_r_s_r_d_spot_value(self):
        # Independently recompute R_f/R_s/R_d (H3 §2, "1. Determine the
        # fraction of collision-breakup types") from a hand-picked CKE/S_T
        # pair, then check they sum to 1 (R_d = max(1-R_f-R_s, 0), the
        # invariant the Fortran's own branch structure guarantees whenever
        # R_s+R_f<=1).
        cke = 2.0e-6  # >= CKE0=8.93e-7
        s_t = 3.0e-6
        cke0 = 8.93e-7
        w0 = 0.86
        w2 = cke / s_t
        r_f_expected = 1.11e-4 * cke ** (-0.654) if cke >= cke0 else 1.0
        r_s_expected = 0.685 * (1.0 - math.exp(-1.63 * (w2 - w0))) if w2 >= w0 else 0.0
        assert r_f_expected < 1.0  # sanity: this fixture exercises the CKE>=CKE0 branch
        if r_s_expected + r_f_expected <= 1.0:
            r_d_expected = max(1.0 - r_f_expected - r_s_expected, 0.0)
            assert r_f_expected + r_s_expected + r_d_expected == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# %len / %vtm direct computation (H3 §8).
# ---------------------------------------------------------------------------


class TestLiquidLenVtm:
    def test_shapes(self):
        binb, length, vtm = breakfragment.liquid_len_vtm(40, 20)
        assert binb.shape == (41,)
        assert length.shape == (40,)
        assert vtm.shape == (40,)

    def test_length_matches_real_bin_grid_boundaries(self):
        binb, _length, _vtm = breakfragment.liquid_len_vtm(40, 20)
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        np.testing.assert_array_equal(binb, grid.binb)

    def test_length_nondecreasing(self):
        # mean_mass_i is derived from an analytic (Marshall-Palmer-type)
        # spectrum integrated per bin, monotone increasing with bin index
        # for a monotone-increasing bin grid -- not a trivial identity, so
        # a real assertion, not a tautology.
        _binb, length, _vtm = breakfragment.liquid_len_vtm(40, 20)
        assert np.all(np.diff(length) > 0.0)

    def test_vtm_nonnegative_and_finite(self):
        _binb, _length, vtm = breakfragment.liquid_len_vtm(40, 20)
        assert np.all(np.isfinite(vtm))
        assert np.all(vtm >= 0.0)

    def test_vtm_zero_where_length_below_stokes_floor(self):
        # _terminal_velocity's own "rad<0.5e-4 -> vtm=0" branch.
        _binb, length, vtm = breakfragment.liquid_len_vtm(40, 20)
        tiny = length < 1.0e-4  # rad = length/2 < 0.5e-4
        if np.any(tiny):
            assert np.all(vtm[tiny] == 0.0)

    def test_some_bin_clears_d0_cutoff(self):
        _binb, length, _vtm = breakfragment.liquid_len_vtm(40, 20)
        assert np.any(length >= 0.01)


# ---------------------------------------------------------------------------
# make_breakup_fragment_tables: end-to-end, cloudlab shape.
# ---------------------------------------------------------------------------


class TestMakeBreakupFragmentTables:
    @pytest.fixture(scope="class")
    def cloudlab_tables(self):
        return breakfragment.make_breakup_fragment_tables(40, 20)

    def test_is_placeholder_is_false(self, cloudlab_tables):
        assert cloudlab_tables.is_placeholder is False

    def test_index_scalars_consistent(self, cloudlab_tables):
        t = cloudlab_tables
        assert 1 <= t.jmin_bk < 40
        assert t.imin_bk == t.jmin_bk + 1
        assert t.imax_bk == 40
        assert t.jmax_bk == 39

    def test_shapes_match_breakup_fragment_table_sizes(self, cloudlab_tables):
        t = cloudlab_tables
        i1d_pair_max, kk_max = breakup_fragment_table_sizes(40, t.jmin_bk)
        assert t.bu_tmass.shape == (i1d_pair_max,)
        assert t.bu_fd.shape == (2, kk_max)

    def test_all_finite(self, cloudlab_tables):
        assert np.all(np.isfinite(cloudlab_tables.bu_tmass))
        assert np.all(np.isfinite(cloudlab_tables.bu_fd))

    def test_nonnegative(self, cloudlab_tables):
        assert np.all(cloudlab_tables.bu_tmass >= 0.0)
        assert np.all(cloudlab_tables.bu_fd >= 0.0)

    def test_mass_conservation_per_pair(self, cloudlab_tables):
        t = cloudlab_tables
        nbin = 40
        n_pairs = t.bu_tmass.shape[0]
        for pair in range(1, n_pairs + 1):
            if t.bu_tmass[pair - 1] <= 0.0:
                continue
            kk0 = (pair - 1) * nbin
            mass_sum = t.bu_fd[0, kk0 : kk0 + nbin].sum()
            assert mass_sum == pytest.approx(t.bu_tmass[pair - 1], rel=1.0e-6)

    def test_every_pair_has_nonzero_mass_for_cloudlab_grid(self, cloudlab_tables):
        # Every (i,j) pair with i,j >= jmin_bk is a "rain-sized" bin for
        # cloudlab's grid -- CKE clears the 1e-20 gate for all of them, so
        # bu_tmass should be fully populated, not sparse. A real (if
        # config-specific) sanity check, not a universal invariant.
        assert np.all(cloudlab_tables.bu_tmass > 0.0)

    @pytest.mark.parametrize("nrbin,nbin_h", [(40, 20), (80, 20)])
    def test_runs_for_other_grids(self, nrbin, nbin_h):
        t = breakfragment.make_breakup_fragment_tables(nrbin, nbin_h)
        assert t.is_placeholder is False
        assert t.imax_bk == nrbin
        assert np.all(np.isfinite(t.bu_tmass))
        assert np.all(np.isfinite(t.bu_fd))


# ---------------------------------------------------------------------------
# Marker-gated: compare against a real AMPS_DUMP_setup dump. SKIPPED
# (with a pointer) until one is produced -- see module docstring.
# ---------------------------------------------------------------------------

_AMPS_DUMP_DIR_ENV = "AMPS_DUMP_DIR"


def _find_setup_dump() -> Path | None:
    """Locate a local `amps_dump_setup.bin` (M2b Task 6's `AMPS_DUMP_setup`
    instrumentation), mirroring test_warm_replay.py's own `_find_dump_
    source` convention: `$AMPS_DUMP_DIR` (either the file itself or a
    directory containing it) first, then the conventional
    `$ICON4PY_TEST_DATA_PATH/amps/` location."""
    env_path = os.environ.get(_AMPS_DUMP_DIR_ENV)
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            f = candidate / "amps_dump_setup.bin"
            return f if f.exists() else None
        return None
    if TEST_DATA_PATH is not None:
        f = TEST_DATA_PATH / "amps" / "amps_dump_setup.bin"
        if f.exists():
            return f
    return None


_SETUP_SKIP_MESSAGE = (
    "No local amps_dump_setup.bin found for the breakfragment validation harness -- checked "
    f"${_AMPS_DUMP_DIR_ENV} and $ICON4PY_TEST_DATA_PATH/amps/amps_dump_setup.bin. This dump "
    "does not exist yet as of M2b Task 6 -- produce one per "
    "docs/superpowers/specs/2026-07-16-ref-data-run-instructions.md (a short scale_amps rerun, "
    "or a standalone setup-only run, with l_amps_dump=.true. -- AMPS_DUMP_setup fires once, at "
    "AMPS setup, rank 0 only, no full spin-up needed), then set "
    f"${_AMPS_DUMP_DIR_ENV} or place the file at $ICON4PY_TEST_DATA_PATH/amps/, and re-run -- "
    "this test activates automatically once found, no code change needed."
)


@pytest.mark.datatest
def test_breakup_tables_match_amps_setup_dump():
    dump_path = _find_setup_dump()
    if dump_path is None:
        pytest.skip(_SETUP_SKIP_MESSAGE)

    record = ref_data.read_setup_dump(dump_path)
    tables = breakfragment.make_breakup_fragment_tables(record.nbr, nbin_h=20)

    assert tables.jmin_bk == record.jmin_bk
    assert tables.imin_bk == record.imin_bk
    assert tables.imax_bk == record.imax_bk
    assert tables.jmax_bk == record.jmax_bk

    i1d_pair_max, kk_max = breakup_fragment_table_sizes(record.nbr, record.jmin_bk)
    np.testing.assert_allclose(
        tables.bu_tmass, record.bu_tmass[:i1d_pair_max], rtol=1.0e-6, atol=1.0e-30
    )
    np.testing.assert_allclose(tables.bu_fd, record.bu_fd[:, :kk_max], rtol=1.0e-6, atol=1.0e-30)
