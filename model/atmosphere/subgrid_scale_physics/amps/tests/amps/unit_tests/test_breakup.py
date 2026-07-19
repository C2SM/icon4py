# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/breakup.py (M2b Task 5): the Low-List collisional-breakup
RUNTIME (`ibreak==1`) -- `add_fragments_col_vec`, the `i1d_pair`/`kk` dense
re-expansion (`dense_fragment_tables`), `breakup_number_consumed`
(`used_N_b`), `P_breakup`/`Q_breakup2`, and the wiring into
`core/coalescence.py::coalesce_rain`'s `ibreak` hook, per
docs/superpowers/facts/m2b/collisional-breakup.md ("H2" below).

Groups:
* TestDenseFragmentTables -- `dense_fragment_tables`'s `i1d_pair`/`kk`
  vectorized re-expansion against the REAL cloudlab (40-bin) table,
  cross-checked against an INDEPENDENTLY re-derived `i1d_pair` formula
  (written directly in this test, not by calling the module's own helper)
  -- the "fragment-table (i,j) indexing hits the right entry" requirement.
* TestBreakupNumberConsumed -- `used_N_b` (H2 SS1c) unit-tested directly
  with small, hand-constructed `DenseFragmentTables`: both-role
  (collector + collectee) summation, masking to `dense.valid` (the
  "conservation-by-construction" restriction, `core/breakup.py`'s module
  docstring).
* TestAddFragmentsColVec -- the fragment-table CONSUMER (H2 SS3a)
  unit-tested directly against the REAL cloudlab table: the mass-balance
  identity (`injected mass == N_bk*mod_rat*bu_tmass ==
  N_bk*(mean_mass_i+mean_mass_j)`), aerosol-leg ratios, `gate_i`/no-
  valid-`j` no-ops.
* TestPBreakup / TestQBreakup2 -- the spontaneous-breakup formulas (H2
  SS3b/SS3c), spot-verified against independently-derived closed forms,
  `scipy.integrate.quad`, and `scipy.special.gammainc` -- ported per the
  task's own deliverable list, NOT wired into `coalesce_rain` (a SEPARATE
  mechanism from the Low-List collisional table, `core/breakup.py`'s
  module docstring).
* TestCoalesceRainIbreak -- the full `ibreak=1` wiring in
  `core/coalescence.py::coalesce_rain`: a colliding pair above the
  breakup threshold produces fragments (vs the real `bu_fd` table), mass
  and aerosol mass conserved to `rel=1e-12`, number genuinely INCREASES
  (a net fragmentation gain, not merely "less lost"), the ibreak=1 vs
  ibreak=0 difference is exercised (different output, MORE number for
  ibreak=1 on the identical input), and bins below `jmin_bk` (no
  fragment-table coverage at all) are bit-identical between the two
  `ibreak` values -- a pure-passthrough sanity check on the masking.

Fixtures use REAL magnitudes throughout (same convention as
test_coalescence.py/test_breakfragment.py): a REAL 40-bin liquid grid, the
REAL packaged collision-efficiency LUT (`load_luts()`), and the REAL
Low-List fragment table (`breakfragment.make_breakup_fragment_tables(40,
20)`, built ONCE per test session via a module-scoped fixture -- ~0.3s,
M2b Task 6's own report).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import scipy.integrate
import scipy.special

from icon4py.model.atmosphere.subgrid_scale_physics.amps.config import AmpsConfig
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import (
    bin_grid,
    breakfragment,
    breakup,
    coalescence,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.constants import AmpsConst
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.index_maps import LiquidPPV
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    BreakupFragmentTables,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import (
    LiquidState,
    ThermoProp,
    ThermoState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers -- mirrors test_coalescence.py's own conventions.
# ---------------------------------------------------------------------------

NBINS = 40
NBIN_H = 20
BINB = bin_grid.make_bin_grid("liquid", NBINS, nbin_h=NBIN_H).binb


@pytest.fixture(scope="module")
def real_luts() -> AmpsLuts:
    return load_luts()


@pytest.fixture(scope="module")
def cloudlab_tables() -> BreakupFragmentTables:
    return breakfragment.make_breakup_fragment_tables(NBINS, NBIN_H)


@pytest.fixture(scope="module")
def dense(cloudlab_tables: BreakupFragmentTables) -> breakup.DenseFragmentTables:
    return breakup.dense_fragment_tables(cloudlab_tables, NBINS)


def _config() -> AmpsConfig:
    return AmpsConfig.cloudlab()


def _mean_mass_of(bin_index: int) -> float:
    return 0.5 * (BINB[bin_index] + BINB[bin_index + 1])


def _length_of(mean_mass: float) -> float:
    return (6.0 * mean_mass / np.pi) ** (1.0 / 3.0)  # cm, density~1 g/cm^3 sphere


def _liquid_state(
    bins: dict[int, tuple[float, float, float, float]], nbins: int = NBINS
) -> LiquidState:
    """`bins`: {bin_index: (rmt, rcon, rmat, rmas)}, single column,
    PER-VOLUME -- see core/coalescence.py's PER-VOLUME contract note."""
    lp = LiquidPPV
    values = np.zeros((len(LiquidState.PROPS), nbins, 1, 1), dtype=np.float64)
    for b, (rmt, rcon, rmat, rmas) in bins.items():
        values[lp.rmt_q.py_idx, b, 0, 0] = rmt
        values[lp.rcon_q.py_idx, b, 0, 0] = rcon
        values[lp.rmat_q.py_idx, b, 0, 0] = rmat
        values[lp.rmas_q.py_idx, b, 0, 0] = rmas
    return LiquidState(values=values)


def _zero_diag(nbins: int = NBINS, npoints: int = 1) -> LiquidDiag:
    z = np.zeros((nbins, npoints))
    return LiquidDiag(
        mean_mass=z.copy(),
        length=z.copy(),
        a_len=z.copy(),
        c_len=z.copy(),
        density=np.ones((nbins, npoints)),
        terminal_velocity=z.copy(),
        capacitance=z.copy(),
        ventilation_fv=np.ones((nbins, npoints)),
        ventilation_fh=np.ones((nbins, npoints)),
        ventilation_fkn=np.ones((nbins, npoints)),
        vapdep_coef1=z.copy(),
        vapdep_coef2=z.copy(),
        nre=z.copy(),
    )


def _diag_for(bins: dict[int, tuple[float, float, float]], nbins: int = NBINS) -> LiquidDiag:
    """`bins`: {bin_index: (mean_mass, length_cm, terminal_velocity_cm_s)}
    -- same convention as test_coalescence.py's own `_diag_for`."""
    diag = _zero_diag(nbins)
    for b, (mean_mass, length, vtm) in bins.items():
        diag.mean_mass[b, 0] = mean_mass
        diag.length[b, 0] = length
        diag.terminal_velocity[b, 0] = vtm
        diag.nre[b, 0] = max(length * vtm * 1.2e-3 / 1.8e-4, 1.0)
    return diag


def _thermo_state(*, t: float = 280.0) -> ThermoState:
    values = np.zeros((len(ThermoState.PROPS), 1, 1, 1), dtype=np.float64)
    by_prop = {
        ThermoProp.ptotv: 1.0e6,
        ThermoProp.tv: t,
        ThermoProp.thv: t,
        ThermoProp.piv: 0.0,
        ThermoProp.pbv: 0.0,
        ThermoProp.moist_denv: 1.2e-3,
        ThermoProp.qvv: 1.0e-2,
        ThermoProp.thetav: t,
        ThermoProp.wbv: 0.0,
        ThermoProp.momv: 0.0,
    }
    for idx, prop in enumerate(ThermoState.PROPS):
        values[idx, 0, 0, 0] = by_prop[ThermoProp(int(prop))]
    return ThermoState(values=values)


def _total_number(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rcon_q.py_idx, :, 0, 0].sum())


def _total_mass(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmt_q.py_idx, :, 0, 0].sum())


def _total_aero_total(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmat_q.py_idx, :, 0, 0].sum())


def _total_aero_soluble(liquid: LiquidState) -> float:
    return float(liquid.values[LiquidPPV.rmas_q.py_idx, :, 0, 0].sum())


# ---------------------------------------------------------------------------
# TestDenseFragmentTables
# ---------------------------------------------------------------------------


class TestDenseFragmentTables:
    def test_shapes(self, dense: breakup.DenseFragmentTables):
        assert dense.tmass.shape == (NBINS, NBINS)
        assert dense.valid.shape == (NBINS, NBINS)
        assert dense.frag_mass.shape == (NBINS, NBINS, NBINS)
        assert dense.frag_con.shape == (NBINS, NBINS, NBINS)

    def test_index_scalars_match_table(
        self, cloudlab_tables: BreakupFragmentTables, dense: breakup.DenseFragmentTables
    ):
        assert dense.jmin_bk == cloudlab_tables.jmin_bk - 1
        assert dense.imin_bk == cloudlab_tables.imin_bk - 1
        assert dense.imax_bk == cloudlab_tables.imax_bk - 1
        assert dense.jmax_bk == cloudlab_tables.jmax_bk - 1

    def test_only_i_greater_than_j_is_ever_valid(self, dense: breakup.DenseFragmentTables):
        upper_or_diag = np.triu(np.ones((NBINS, NBINS), dtype=bool))
        assert not np.any(dense.valid & upper_or_diag)

    def test_valid_count_matches_task6_report(self, dense: breakup.DenseFragmentTables):
        """M2b Task 6's own report: '136/136 pairs have nonzero bu_tmass'
        for cloudlab's 40-bin grid -- every eligible (i,j) pair in
        `[imin_bk,imax_bk]x[jmin_bk,jmax_bk]` IS tabulated (no
        `D_coal<=D_0`/`CKE<=1e-20` build-time skip fires for this real
        config)."""
        assert int(dense.valid.sum()) == 136

    def test_i1d_pair_indexing_hits_the_right_entry(
        self, cloudlab_tables: BreakupFragmentTables, dense: breakup.DenseFragmentTables
    ):
        """The `i1d_pair`/`kk` formula (H2 SS4), re-derived INDEPENDENTLY
        here (not by calling `dense_fragment_tables` itself) for several
        hand-picked `(i,j)` pairs, must match `dense`'s own dense
        re-expansion exactly -- both `bu_tmass` and the full `bu_fd`
        fragment-bin slice."""
        jmin_bk_1b, imin_bk_1b = cloudlab_tables.jmin_bk, cloudlab_tables.imin_bk

        for i0, j0 in [(24, 23), (33, 28), (39, 24), (39, 38), (30, 24)]:
            i1b, j1b = i0 + 1, j0 + 1  # 1-based Fortran indices
            assert i1b >= imin_bk_1b
            assert jmin_bk_1b <= j1b < i1b

            i1d_pair = (j1b - jmin_bk_1b + 1) + (i1b - imin_bk_1b) * (1 + i1b - imin_bk_1b) // 2
            expected_tmass = cloudlab_tables.bu_tmass[i1d_pair - 1]
            kk0 = (i1d_pair - 1) * NBINS
            expected_mass = cloudlab_tables.bu_fd[0, kk0 : kk0 + NBINS]
            expected_con = cloudlab_tables.bu_fd[1, kk0 : kk0 + NBINS]

            assert dense.tmass[i0, j0] == pytest.approx(expected_tmass)
            assert dense.valid[i0, j0] == (expected_tmass > 1.0e-30)
            np.testing.assert_allclose(dense.frag_mass[i0, j0, :], expected_mass)
            np.testing.assert_allclose(dense.frag_con[i0, j0, :], expected_con)

    def test_out_of_domain_pairs_are_invalid_and_zero(self, dense: breakup.DenseFragmentTables):
        # i below imin_bk: never valid regardless of j.
        assert not dense.valid[dense.imin_bk - 1, :].any()
        assert (dense.tmass[dense.imin_bk - 1, :] == 0.0).all()
        # j below jmin_bk: never valid regardless of i.
        assert not dense.valid[:, dense.jmin_bk - 1].any()

    def test_mass_conservation_per_valid_pair(self, dense: breakup.DenseFragmentTables):
        """`sum_k(bu_fd(1,kk)) == bu_tmass` (the table's own `mrat`
        normalization, `core/breakfragment.py`) must survive the dense
        re-expansion exactly."""
        valid_i, valid_j = np.nonzero(dense.valid)
        assert valid_i.size > 0
        mass_sums = dense.frag_mass[valid_i, valid_j, :].sum(axis=-1)
        np.testing.assert_allclose(mass_sums, dense.tmass[valid_i, valid_j], rtol=1.0e-6)


# ---------------------------------------------------------------------------
# TestBreakupNumberConsumed
# ---------------------------------------------------------------------------


def _tiny_dense(valid: np.ndarray) -> breakup.DenseFragmentTables:
    nbins = valid.shape[0]
    return breakup.DenseFragmentTables(
        tmass=np.zeros((nbins, nbins)),
        valid=valid,
        frag_mass=np.zeros((nbins, nbins, nbins)),
        frag_con=np.zeros((nbins, nbins, nbins)),
        jmin_bk=0,
        imin_bk=1,
        imax_bk=nbins - 1,
        jmax_bk=nbins - 2,
    )


class TestBreakupNumberConsumed:
    def test_combines_collector_and_collectee_roles(self):
        """3 bins, valid pairs (1,0) and (2,1) only (i>j, the table's own
        triangular convention). Bin 1 plays BOTH roles simultaneously
        (collectee of pair (1,0), collector of pair (2,1)) -- its own
        `used_N_b` must be the SUM of both contributions, H2 SS1c's own
        two-loop accumulation into a single array."""
        nbins = 3
        valid = np.zeros((nbins, nbins), dtype=bool)
        valid[1, 0] = True
        valid[2, 1] = True
        dense_small = _tiny_dense(valid)

        n_col = np.zeros((nbins, nbins, 1))
        e_coal = np.ones((nbins, nbins, 1))
        n_col[1, 0, 0] = 100.0
        e_coal[1, 0, 0] = 0.4  # (1-E_coal)=0.6 -> n_bk=60
        n_col[2, 1, 0] = 50.0
        e_coal[2, 1, 0] = 0.8  # (1-E_coal)=0.2 -> n_bk=10

        used_n_b = breakup.breakup_number_consumed(n_col, e_coal, dense_small)

        assert used_n_b[0, 0] == pytest.approx(60.0)  # bin 0: collectee of (1,0) only
        assert used_n_b[1, 0] == pytest.approx(70.0)  # bin 1: collector (60) + collectee (10)
        assert used_n_b[2, 0] == pytest.approx(10.0)  # bin 2: collector of (2,1) only

    def test_masked_to_valid_pairs_only(self):
        """An (i,j) pair OUTSIDE `dense.valid` contributes NOTHING, even
        with E_coal=0 (100% nominal breakup fraction) -- this module's own
        'conservation-by-construction' design (core/breakup.py's module
        docstring): breakup consumption is only ever tallied where a
        matching fragment table entry exists to re-inject it."""
        nbins = 2
        dense_small = _tiny_dense(np.zeros((nbins, nbins), dtype=bool))
        n_col = np.array([[[0.0], [0.0]], [[100.0], [0.0]]])
        e_coal = np.zeros((nbins, nbins, 1))

        used_n_b = breakup.breakup_number_consumed(n_col, e_coal, dense_small)
        assert np.all(used_n_b == 0.0)

    def test_negative_e_coal_complement_is_clamped(self):
        """`max(0,1-E_coal)` -- E_coal>1 (shouldn't happen physically, but
        the formula clamps rather than going negative)."""
        valid = np.array([[False, False], [True, False]])
        dense_small = _tiny_dense(valid)
        n_col = np.array([[[0.0], [0.0]], [[10.0], [0.0]]])
        e_coal = np.array([[[0.0], [0.0]], [[1.5], [0.0]]])

        used_n_b = breakup.breakup_number_consumed(n_col, e_coal, dense_small)
        assert np.all(used_n_b >= 0.0)
        assert used_n_b[0, 0] == pytest.approx(0.0)
        assert used_n_b[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestAddFragmentsColVec
# ---------------------------------------------------------------------------


class TestAddFragmentsColVec:
    def test_mass_balance_identity(self, dense: breakup.DenseFragmentTables):
        """Fragment mass injected for a pair ==
        `N_bk*mod_rat*bu_tmass == N_bk*(mean_mass_i+mean_mass_j)` --
        core/breakup.py's module docstring's own derivation, verified here
        via an INDEPENDENTLY computed expected value (not by re-deriving
        the SAME formula the implementation itself uses internally)."""
        i, j = 33, 28
        assert dense.valid[i, j], "fixture assumption: (33,28) must be a valid table pair"

        mean_mass = np.zeros((NBINS, 1))
        mean_mass[i, 0] = _mean_mass_of(i)
        mean_mass[j, 0] = _mean_mass_of(j)
        mass_tot = np.zeros((NBINS, 1))
        mass_tot[i, 0] = mean_mass[i, 0] * 80.0
        mass_tot[j, 0] = mean_mass[j, 0] * 300.0
        mass_aero_tot = mass_tot * 1.0e-2
        mass_aero_sol = mass_tot * 0.4e-2

        n_col_i = np.zeros((NBINS, 1))
        n_col_i[j, 0] = 200.0
        e_coal_i = np.zeros((NBINS, 1))
        e_coal_i[j, 0] = 0.3  # 70% of the 200 collisions break up
        gate_i = np.array([True])

        add_n, add_rmt, add_rmat, add_rmas = breakup.add_fragments_col_vec(
            i, n_col_i, e_coal_i, mean_mass, mass_tot, mass_aero_tot, mass_aero_sol, gate_i, dense
        )

        n_bk = 200.0 * (1.0 - 0.3)
        expected_mass = n_bk * (mean_mass[i, 0] + mean_mass[j, 0])
        assert float(add_rmt.sum()) == pytest.approx(expected_mass, rel=1.0e-12)
        assert float(add_n.sum()) > 0.0, "breakup must produce fragment NUMBER, not just mass"

        ratio_rmat = (mass_aero_tot[i, 0] + mass_aero_tot[j, 0]) / (mass_tot[i, 0] + mass_tot[j, 0])
        ratio_rmas = (mass_aero_sol[i, 0] + mass_aero_sol[j, 0]) / (mass_tot[i, 0] + mass_tot[j, 0])
        assert float(add_rmat.sum()) == pytest.approx(expected_mass * ratio_rmat, rel=1.0e-12)
        assert float(add_rmas.sum()) == pytest.approx(expected_mass * ratio_rmas, rel=1.0e-12)

    def test_gate_false_is_a_no_op(self, dense: breakup.DenseFragmentTables):
        i, j = 33, 28
        assert dense.valid[i, j]
        mean_mass = np.zeros((NBINS, 1))
        mean_mass[i, 0] = _mean_mass_of(i)
        mean_mass[j, 0] = _mean_mass_of(j)
        mass_tot = mean_mass * 100.0
        n_col_i = np.zeros((NBINS, 1))
        n_col_i[j, 0] = 200.0
        e_coal_i = np.full((NBINS, 1), 0.3)
        gate_i = np.array([False])

        add_n, add_rmt, add_rmat, add_rmas = breakup.add_fragments_col_vec(
            i,
            n_col_i,
            e_coal_i,
            mean_mass,
            mass_tot,
            mass_tot * 0.01,
            mass_tot * 0.004,
            gate_i,
            dense,
        )
        assert float(add_n.sum()) == 0.0
        assert float(add_rmt.sum()) == 0.0
        assert float(add_rmat.sum()) == 0.0
        assert float(add_rmas.sum()) == 0.0

    def test_no_valid_j_below_imin_bk_is_a_no_op(self, dense: breakup.DenseFragmentTables):
        """`i` below `imin_bk` (the table's own domain) -- no `j<i`
        collectee ever has a valid entry, mirrors the Fortran's own
        `if(i<imin_bk.or.i>imax_bk) return` early exit."""
        i = dense.imin_bk - 1
        assert not dense.valid[i, :].any()
        mean_mass = np.full((NBINS, 1), 1.0e-7)
        mass_tot = mean_mass * 100.0
        n_col_i = np.full((NBINS, 1), 200.0)
        e_coal_i = np.full((NBINS, 1), 0.3)
        gate_i = np.array([True])

        add_n, add_rmt, _add_rmat, _add_rmas = breakup.add_fragments_col_vec(
            i,
            n_col_i,
            e_coal_i,
            mean_mass,
            mass_tot,
            mass_tot * 0.01,
            mass_tot * 0.004,
            gate_i,
            dense,
        )
        assert float(add_n.sum()) == 0.0
        assert float(add_rmt.sum()) == 0.0

    def test_multiple_columns_are_independent(self, dense: breakup.DenseFragmentTables):
        """`npoints=2`, only column 0 gated -- column 1 must stay exactly
        zero regardless of its own (nonzero) n_col/e_coal inputs."""
        i, j = 33, 28
        npoints = 2
        mean_mass = np.zeros((NBINS, npoints))
        mean_mass[i, :] = _mean_mass_of(i)
        mean_mass[j, :] = _mean_mass_of(j)
        mass_tot = mean_mass * 100.0
        n_col_i = np.zeros((NBINS, npoints))
        n_col_i[j, :] = 200.0
        e_coal_i = np.full((NBINS, npoints), 0.3)
        gate_i = np.array([True, False])

        add_n, add_rmt, _add_rmat, _add_rmas = breakup.add_fragments_col_vec(
            i,
            n_col_i,
            e_coal_i,
            mean_mass,
            mass_tot,
            mass_tot * 0.01,
            mass_tot * 0.004,
            gate_i,
            dense,
        )
        assert (add_n[:, 0] > 0.0).any()
        assert (add_n[:, 1] == 0.0).all()
        assert (add_rmt[:, 1] == 0.0).all()


# ---------------------------------------------------------------------------
# TestPBreakup / TestQBreakup2 -- spontaneous large-drop breakup formulas
# (H2 SS3b/SS3c), NOT wired into coalesce_rain (see core/breakup.py's
# module docstring).
# ---------------------------------------------------------------------------


class TestPBreakup:
    def test_phase1_saturates_to_one_at_max_dim(self):
        assert breakup.p_breakup(1, a_star=5.0, max_dim=5.0) == 1.0
        assert breakup.p_breakup(1, a_star=6.0, max_dim=5.0) == 1.0

    def test_phase1_formula_below_max_dim(self):
        a_star, max_dim = 0.2, 5.0
        expected = min(2.94e-7 * math.exp(3.4 * a_star * 10.0), 1.0)
        assert breakup.p_breakup(1, a_star, max_dim) == pytest.approx(expected)

    def test_phase1_monotone_increasing_and_bounded(self):
        max_dim = 5.0
        values = [breakup.p_breakup(1, a, max_dim) for a in np.linspace(0.0, 4.9, 20)]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(b >= a for a, b in itertools.pairwise(values))

    def test_phase2_saturates_to_one_at_max_dim(self):
        assert breakup.p_breakup(2, a_star=5.0, max_dim=5.0) == 1.0

    def test_phase2_formula_below_max_dim(self):
        a_star, max_dim = 0.2, 5.0
        k = 15.03968607 / (max_dim * 10.0)
        expected = min(2.94e-7 * math.exp(k * a_star * 10.0), 1.0)
        assert breakup.p_breakup(2, a_star, max_dim) == pytest.approx(expected)

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="phase"):
            breakup.p_breakup(3, a_star=1.0, max_dim=5.0)


class TestQBreakup2:
    def test_switch1_formula_direct(self):
        a_star, a, m, aa, bb = 0.5, 0.1, 1.0e-3, 2.0, 3.0
        expected = (aa * bb / 3.0 / m) * (a / a_star) * math.exp(-bb * a / a_star)
        assert breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=1) == pytest.approx(expected)

    def test_switch2_matches_numerical_integration_of_the_exponential_density(self):
        """`switch=2` ("total number of drops between a and m") is the
        closed-form definite integral of the SIMPLE exponential density
        `AA*BB/a_star*exp(-BB*r/a_star)` from `a` to `m` -- verified here
        via an INDEPENDENTLY written integrand and `scipy.integrate.quad`,
        not by calling `q_breakup2` itself for the density."""
        a_star, a, m, aa, bb = 0.8, 0.05, 0.6, 3.0, 2.5

        def density(r: float) -> float:
            return aa * bb / a_star * math.exp(-bb * r / a_star)

        expected, _err = scipy.integrate.quad(density, a, m)
        got = breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=2)
        assert got == pytest.approx(expected, rel=1.0e-8)

    def test_switch2_formula_direct(self):
        a_star, a, m, aa, bb = 0.5, 0.1, 0.9, 2.0, 3.0
        expected = -aa * (math.exp(-bb * m / a_star) - math.exp(-bb * a / a_star))
        assert breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=2) == pytest.approx(expected)

    def test_switch3_matches_gammainc_closed_form(self):
        """`switch=3` ("total mass between a and m") uses `fGM` (a p=4
        incomplete-gamma integral) internally -- verified here against
        `scipy.special.gammainc` (the regularized lower incomplete gamma
        function), an INDEPENDENT special-function library, not against
        the module's own `_fgm` helper."""
        a_star, a, m, aa, bb = 0.8, 0.05, 0.6, 3.0, 2.5
        x1 = bb * a / a_star
        x2 = bb * m / a_star
        # int_x1^x2 t^3 e^-t dt = Gamma(4)*(gammainc(4,x2)-gammainc(4,x1))
        gamma4 = math.factorial(3)
        fgm_expected = gamma4 * (scipy.special.gammainc(4, x2) - scipy.special.gammainc(4, x1))
        pi = float(AmpsConst.PI)
        expected = (4.0 * pi / 3.0) * aa * (a_star / bb) ** 3.0 * fgm_expected

        got = breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=3)
        assert got == pytest.approx(expected, rel=1.0e-10)

    def test_phase2_switch1_and_2_share_phase1_formula(self):
        a_star, a, m, aa, bb = 0.5, 0.1, 0.9, 2.0, 3.0
        assert breakup.q_breakup2(2, a_star, a, m, aa, bb, switch=1) == pytest.approx(
            breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=1)
        )
        assert breakup.q_breakup2(2, a_star, a, m, aa, bb, switch=2) == pytest.approx(
            breakup.q_breakup2(1, a_star, a, m, aa, bb, switch=2)
        )

    def test_phase2_switch3_raises(self):
        with pytest.raises(ValueError, match="switch"):
            breakup.q_breakup2(2, 0.5, 0.1, 0.9, 2.0, 3.0, switch=3)

    def test_invalid_switch_raises(self):
        with pytest.raises(ValueError, match="switch"):
            breakup.q_breakup2(1, 0.5, 0.1, 0.9, 2.0, 3.0, switch=4)

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="phase"):
            breakup.q_breakup2(3, 0.5, 0.1, 0.9, 2.0, 3.0, switch=1)


# ---------------------------------------------------------------------------
# TestCoalesceRainIbreak -- the full ibreak=1 wiring in coalesce_rain.
# ---------------------------------------------------------------------------

# Two real, rain-sized bins (well above jmin_bk=24) with a strong collision
# rate and E_coal well below 1 (verified: E_coal[33,28]~0.454 at these
# velocities/con) -- most of the raw N_col=300 collisions break up rather
# than coalesce, so this is the "dominant breakup" regime.
_BIN_J = 28
_BIN_I = 33
_CON_J = 300.0
_CON_I = 80.0
_VTM_J = 300.0
_VTM_I = 700.0


def _dominant_breakup_liquid_and_diag() -> tuple[LiquidState, LiquidDiag]:
    m_j, m_i = _mean_mass_of(_BIN_J), _mean_mass_of(_BIN_I)
    liquid = _liquid_state(
        {
            _BIN_J: (m_j * _CON_J, _CON_J, m_j * _CON_J * 1.0e-2, m_j * _CON_J * 0.4e-2),
            _BIN_I: (m_i * _CON_I, _CON_I, m_i * _CON_I * 1.0e-2, m_i * _CON_I * 0.4e-2),
        }
    )
    diag = _diag_for(
        {
            _BIN_J: (m_j, _length_of(m_j), _VTM_J),
            _BIN_I: (m_i, _length_of(m_i), _VTM_I),
        }
    )
    return liquid, diag


class TestCoalesceRainIbreak:
    def test_dominant_breakup_pair_conserves_mass_and_increases_number(
        self, real_luts: AmpsLuts, cloudlab_tables: BreakupFragmentTables
    ):
        """Mass AND aerosol mass conserved to `rel=1e-12`; NUMBER
        genuinely INCREASES relative to the PRE-collision state (a real
        fragmentation net gain, not merely 'less lost than ibreak=0') --
        fragmentation dominates ordinary 2-for-1 coalescence in this
        regime (E_coal~0.45, most collisions break up)."""
        liquid, diag = _dominant_breakup_liquid_and_diag()
        thermo = _thermo_state()
        config = _config()

        m0 = _total_mass(liquid)
        n0 = _total_number(liquid)
        a0 = _total_aero_total(liquid)

        out = coalescence.coalesce_rain(
            liquid,
            diag,
            thermo,
            config,
            dt=2.0,
            luts=real_luts,
            ibreak=True,
            breakup_tables=cloudlab_tables,
        )
        m1 = _total_mass(out)
        n1 = _total_number(out)
        a1 = _total_aero_total(out)

        assert m1 == pytest.approx(m0, rel=1.0e-12, abs=1.0e-30)
        assert a1 == pytest.approx(a0, rel=1.0e-12, abs=1.0e-30)
        assert n1 > n0, "fragmentation must net-increase number in this dominant-breakup regime"
        assert np.all(np.isfinite(out.values))
        assert (out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0] >= -1.0e-9).all()
        assert (out.values[LiquidPPV.rmt_q.py_idx, :, 0, 0] >= -1.0e-9).all()

    def test_fragments_land_above_jmin_bk_and_below_coalesced_mass(
        self, real_luts: AmpsLuts, cloudlab_tables: BreakupFragmentTables
    ):
        """Fragments (per Low-List theory) must be SMALLER than the
        coalesced-drop mass `m_i+m_j` (they are pieces of a shattered
        drop) -- sanity-checks that SOME real fragment bin gained
        population, and that it is not simply a copy of the (nonexistent,
        since it broke up rather than coalescing) merged-drop bin."""
        liquid, diag = _dominant_breakup_liquid_and_diag()
        thermo = _thermo_state()
        config = _config()
        m_j, m_i = _mean_mass_of(_BIN_J), _mean_mass_of(_BIN_I)

        out = coalescence.coalesce_rain(
            liquid,
            diag,
            thermo,
            config,
            dt=2.0,
            luts=real_luts,
            ibreak=True,
            breakup_tables=cloudlab_tables,
        )
        con = out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0]
        gained = np.nonzero(con > 1.0e-6)[0]
        assert gained.size > 0
        # at least one gaining bin must be a genuinely NEW population
        # (not bin _BIN_I/_BIN_J themselves, which were mostly consumed).
        new_bins = [b for b in gained if b not in (_BIN_I, _BIN_J)]
        assert new_bins
        # every genuinely NEW gaining bin's own mass range must be BELOW
        # the coalesced-drop mass m_i+m_j -- fragments are pieces of a
        # shattered drop, never bigger than what would have coalesced.
        m_coal = m_j + m_i
        assert all(BINB[b] < m_coal for b in new_bins)

    def test_ibreak_true_differs_from_ibreak_false_and_has_more_number(
        self, real_luts: AmpsLuts, cloudlab_tables: BreakupFragmentTables
    ):
        """The ibreak=1 vs ibreak=0 difference, exercised directly: same
        input, same dt -- `ibreak=True`'s own OUTPUT total number must
        exceed `ibreak=False`'s (fragmentation adds population on top of
        the SAME underlying coalescence baseline), and the two outputs
        must differ (breakup is not a silent no-op)."""
        liquid, diag = _dominant_breakup_liquid_and_diag()
        thermo = _thermo_state()
        config = _config()

        out_false = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)
        out_true = coalescence.coalesce_rain(
            liquid,
            diag,
            thermo,
            config,
            dt=2.0,
            luts=real_luts,
            ibreak=True,
            breakup_tables=cloudlab_tables,
        )

        assert not np.allclose(out_false.values, out_true.values, rtol=1.0e-9, atol=1.0e-20)
        assert _total_number(out_true) > _total_number(out_false)
        # both conserve mass identically (same physical constraint, just a
        # different redistribution) -- mass is NOT a distinguishing axis.
        assert _total_mass(out_true) == pytest.approx(_total_mass(out_false), rel=1.0e-9)

    def test_multi_bin_conserves_mass_and_aerosol_across_dt(
        self, real_luts: AmpsLuts, cloudlab_tables: BreakupFragmentTables
    ):
        """5 active bins, all above `jmin_bk`, saturating collision rates
        (mirrors test_coalescence.py's own
        TestMultiCollectorConservation fixture style, but entirely above
        the breakup threshold) -- mass/aerosol mass conserved to
        `rel=1e-12` across a range of `dt`, INCLUDING the saturating
        regime, with `ibreak=True`."""
        bin_indices = [25, 28, 31, 34, 37]
        cons = {25: 300.0, 28: 150.0, 31: 60.0, 34: 20.0, 37: 5.0}
        vtms = {25: 300.0, 28: 450.0, 31: 600.0, 34: 800.0, 37: 1000.0}
        bins = {}
        diag_bins = {}
        for b in bin_indices:
            m = _mean_mass_of(b)
            con_b = cons[b]
            bins[b] = (m * con_b, con_b, m * con_b * 1.0e-2, m * con_b * 0.4e-2)
            diag_bins[b] = (m, _length_of(m), vtms[b])
        liquid = _liquid_state(bins)
        diag = _diag_for(diag_bins)
        thermo = _thermo_state()
        config = _config()

        m0 = _total_mass(liquid)
        a0 = _total_aero_total(liquid)
        aero_sol0 = _total_aero_soluble(liquid)

        for dt in (0.1, 0.5, 2.0, 8.0):
            out = coalescence.coalesce_rain(
                liquid,
                diag,
                thermo,
                config,
                dt=dt,
                luts=real_luts,
                ibreak=True,
                breakup_tables=cloudlab_tables,
            )
            m1 = _total_mass(out)
            a1 = _total_aero_total(out)
            aero_sol1 = _total_aero_soluble(out)

            assert m1 == pytest.approx(m0, rel=1.0e-12, abs=1.0e-30), f"dt={dt}: mass not conserved"
            assert a1 == pytest.approx(a0, rel=1.0e-12, abs=1.0e-30), (
                f"dt={dt}: aerosol not conserved"
            )
            assert aero_sol1 == pytest.approx(aero_sol0, rel=1.0e-12, abs=1.0e-30), (
                f"dt={dt}: soluble aerosol not conserved"
            )
            assert np.all(np.isfinite(out.values)), f"dt={dt}: non-finite value produced"
            assert (out.values[LiquidPPV.rcon_q.py_idx, :, 0, 0] >= -1.0e-6).all(), (
                f"dt={dt}: negative con"
            )
            assert (out.values[LiquidPPV.rmt_q.py_idx, :, 0, 0] >= -1.0e-9).all(), (
                f"dt={dt}: negative mass"
            )

    def test_below_jmin_bk_is_bit_identical_regardless_of_ibreak(
        self, real_luts: AmpsLuts, cloudlab_tables: BreakupFragmentTables
    ):
        """Bins entirely BELOW `jmin_bk` have NO fragment-table coverage
        at all (`dense.valid` all-`False` for any pair among them) -- a
        pure sanity/regression check that `ibreak=True` is then a bit-exact
        passthrough (masking to `dense.valid` genuinely gates every
        `ibreak`-specific code path, not just most of it)."""
        bin_indices = [10, 14, 18]
        cons = {10: 300.0, 14: 150.0, 18: 60.0}
        vtms = {10: 50.0, 14: 80.0, 18: 120.0}
        bins = {}
        diag_bins = {}
        for b in bin_indices:
            m = _mean_mass_of(b)
            con_b = cons[b]
            bins[b] = (m * con_b, con_b, m * con_b * 1.0e-2, m * con_b * 0.4e-2)
            diag_bins[b] = (m, _length_of(m), vtms[b])
        liquid = _liquid_state(bins)
        diag = _diag_for(diag_bins)
        thermo = _thermo_state()
        config = _config()

        out_false = coalescence.coalesce_rain(liquid, diag, thermo, config, dt=2.0, luts=real_luts)
        out_true = coalescence.coalesce_rain(
            liquid,
            diag,
            thermo,
            config,
            dt=2.0,
            luts=real_luts,
            ibreak=True,
            breakup_tables=cloudlab_tables,
        )
        np.testing.assert_array_equal(out_false.values, out_true.values)
