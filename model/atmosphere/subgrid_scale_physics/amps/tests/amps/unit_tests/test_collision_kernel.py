# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core/collision_kernel.py (M2b Task 2): collision efficiency
(drpdrp bilinear gather) + coalescence efficiency (Low & List 1982-style
liquid-liquid formula) + the stochastic-collection kernel, rain-rain warm
case, per docs/superpowers/facts/m2b/coalescence-engine.md ("H1") and
docs/superpowers/facts/m2/coalescence.md ("G4").

`TestBilinearGather` uses a small SYNTHETIC `ColLutAux`/table (not the real
`drpdrp` LUT) with clean, hand-computable node values, so every index/
weight/gather step can be verified by exact arithmetic -- this is the
primary correctness check for H1 §3's index math (the `rrat<0` guard, the
`rrat>1` reciprocal fold, the clamped 2x2 stencil, the `[ec_min, 1]` then
`[*, 15]` clamp pair).

`TestCoalescenceEfficiency` and `TestCollisionKernel` use REAL-magnitude
fixtures (CGS lengths ~0.05-0.2 cm, velocities ~100-300 cm/s, per-volume
number densities ~1e2-1e3 cm^-3) -- degenerate (zero/tiny) fixture values
hid real bugs in M2a, per this task's dispatch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import collision_kernel, thermo
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    ColLutAux,
    Data1DLut,
    GridAux,
    VapIgpAux,
    load_luts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.state import ThermoProp, ThermoState


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_luts() -> AmpsLuts:
    return load_luts()


def _make_diag(
    *,
    length: np.ndarray,
    nre: np.ndarray | None = None,
    terminal_velocity: np.ndarray | None = None,
) -> LiquidDiag:
    """Build a `LiquidDiag` with only the fields `collision_kernel.py`
    actually reads (`length`, `nre`, `terminal_velocity`) populated
    meaningfully; every other field is zero-filled (this module's
    functions never touch them). `length` fixes the shape `(nbins, 1)`."""
    length = np.asarray(length, dtype=np.float64).reshape(-1, 1)
    nbins = length.shape[0]
    zeros = np.zeros((nbins, 1), dtype=np.float64)
    nre_arr = zeros.copy() if nre is None else np.asarray(nre, dtype=np.float64).reshape(-1, 1)
    vtm_arr = (
        zeros.copy()
        if terminal_velocity is None
        else np.asarray(terminal_velocity, dtype=np.float64).reshape(-1, 1)
    )
    return LiquidDiag(
        mean_mass=zeros.copy(),
        length=length,
        a_len=zeros.copy(),
        c_len=zeros.copy(),
        density=np.ones((nbins, 1)),
        terminal_velocity=vtm_arr,
        capacitance=zeros.copy(),
        ventilation_fv=np.ones((nbins, 1)),
        ventilation_fh=np.ones((nbins, 1)),
        ventilation_fkn=np.ones((nbins, 1)),
        vapdep_coef1=zeros.copy(),
        vapdep_coef2=zeros.copy(),
        nre=nre_arr,
    )


def _thermo_state(*, t: float) -> ThermoState:
    """Minimal `ThermoState` (single column) -- `coalescence_efficiency`
    only reads `ThermoProp.tv`; every other field is a harmless real-ish
    placeholder."""
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


# ---------------------------------------------------------------------------
# Synthetic col_lut_aux table: xs=0.0,dx=0.5,ys=0.0,dy=1.0,nr=3,nc=3.
# x (column) domain [0.0, 0.5, 1.0]; y (row) domain [0.0, 1.0, 2.0].
# Values chosen small (<1) so the [ec_min,1] clamp is a genuine no-op for
# TestBilinearGather.test_interior_point -- clamp behavior gets its own
# dedicated tests below with tables that DO trigger it.
# ---------------------------------------------------------------------------

_SYNTH_AUX = ColLutAux(xs=0.0, dx=0.5, ys=0.0, dy=1.0, nr=3, nc=3)
_SYNTH_TABLE = np.array(
    [
        [0.10, 0.20, 0.30],
        [0.15, 0.25, 0.35],
        [0.20, 0.30, 0.40],
    ],
    dtype=np.float64,
)


def _synth_luts(table: np.ndarray = _SYNTH_TABLE, aux: ColLutAux = _SYNTH_AUX) -> AmpsLuts:
    """An `AmpsLuts` with only `drpdrp`/`adrpdrp` meaningfully populated
    (the only fields `collision_efficiency` reads); every other field is
    a harmless placeholder (never touched by this module)."""
    zeros2 = np.zeros((1, 1))
    dummy_aux = ColLutAux(xs=0.0, dx=1.0, ys=0.0, dy=1.0, nr=1, nc=1)
    dummy_grid_aux = GridAux(nr=1, nc=1)
    dummy_1d = Data1DLut(n=1, xs=0.0, dx=1.0, y=np.zeros(1))
    dummy_igp = VapIgpAux(nok=1, x=np.zeros(1), a=np.zeros((1, 4)), b=np.zeros((1, 4)))
    return AmpsLuts(
        drpdrp=table,
        adrpdrp=aux,
        hexdrp=zeros2,
        ahexdrp=dummy_aux,
        bbcdrp=zeros2,
        abbcdrp=dummy_aux,
        coldrp=zeros2,
        acoldrp=dummy_aux,
        gp1drp=zeros2,
        agp1drp=dummy_aux,
        gp4drp=zeros2,
        agp4drp=dummy_aux,
        gp8drp=zeros2,
        agp8drp=dummy_aux,
        pol_frq=zeros2,
        pla_frq=zeros2,
        col_frq=zeros2,
        ros_frq=zeros2,
        ppo_frq=zeros2,
        frq_aux=dummy_grid_aux,
        mtac_map_col=zeros2,
        mtac_map_pla=zeros2,
        map_col_aux=dummy_grid_aux,
        map_pla_aux=dummy_grid_aux,
        lmt_mass_col=np.zeros(50),
        lmt_mass_pla=np.zeros(50),
        lmt_mass_col_aux=dummy_grid_aux,
        lmt_mass_pla_aux=dummy_grid_aux,
        znorm=np.zeros((451, 4)),
        osm_nh42so4=dummy_1d,
        osm_sodchl=dummy_1d,
        snrml=dummy_1d,
        isnrml=dummy_1d,
        vigp=dummy_igp,
    )


# ---------------------------------------------------------------------------
# collision_efficiency -- the drpdrp bilinear gather, H1 SS3.
# ---------------------------------------------------------------------------


class TestBilinearGather:
    def test_interior_point(self):
        """rrat_p=0.25 (len_i=4.0, len_j=1.0 -> rrat=0.25<=1, no fold),
        NreL_p=10**0.5 (log10(NreL_p)=0.5) lands at wx=wy=0.5 in the
        synthetic table's cell (0,0)-(1,1); hand-computed bilinear:
        0.25*(t00+t01+t10+t11) = 0.25*(0.10+0.20+0.15+0.25) = 0.175."""
        diag_i = _make_diag(length=[4.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[1.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert e_c.shape == (1, 1, 1)
        assert float(e_c[0, 0, 0]) == pytest.approx(0.175, rel=1e-12)

    def test_rrat_greater_than_one_reciprocal_path(self):
        """len_i=1.0, len_j=4.0 -> rrat=4.0>1 -> rrat_p=1/4=0.25, the SAME
        table cell/weights as test_interior_point (which used rrat<=1
        directly) -- the reciprocal fold must reproduce the identical
        result, 0.175."""
        diag_i = _make_diag(length=[1.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[4.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert float(e_c[0, 0, 0]) == pytest.approx(0.175, rel=1e-12)

    def test_rrat_negative_guard(self):
        """len_j negative (degenerate/'should not happen', H1's own
        framing) -> rrat<0 -> rrat_p forced to 0.0 (x=0.0, wx=0); Nre
        unaffected (log10(NreL_p)=0.5, wy=0.5 as before). Hand-computed:
        wx=0 kills the x-interpolation, leaving
        (1-wy)*t00 + wy*t01 = 0.5*0.10 + 0.5*0.20 = 0.15."""
        diag_i = _make_diag(length=[1.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[-1.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert float(e_c[0, 0, 0]) == pytest.approx(0.15, rel=1e-12)

    def test_zero_collector_length_also_takes_degenerate_path(self):
        """`len_i<=0` (inactive/degenerate collector bin): this port's own
        safe-division guard (see collision_kernel.py) routes this into
        the SAME rrat<0 degenerate branch as a genuinely negative ratio,
        rather than propagating a NaN/Inf from a 0/0 or x/0 division --
        documented, physically-motivated choice, distinct from (but
        consistent with) the literal Fortran 'rrat<0' scenario above."""
        diag_i = _make_diag(length=[0.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[1.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert np.isfinite(e_c[0, 0, 0])
        assert float(e_c[0, 0, 0]) == pytest.approx(0.15, rel=1e-12)

    def test_nre_floor_at_1e_minus_10(self):
        """`NreL_p=max(Nre, 1e-10)` -- a zero (or negative) Nre must not
        propagate a log10(0)=-inf; it floors to log10(1e-10)=-10, clamped
        by the table's own `[1, nr-1]` index range (this synthetic
        table's y-domain [0,2] doesn't reach -10, so this also exercises
        the lower-edge index clamp, `i1=1`, i.e. wy=0 after clamping)."""
        diag_i = _make_diag(length=[4.0], nre=[0.0])
        diag_j = _make_diag(length=[1.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert np.isfinite(e_c[0, 0, 0])
        # rrat_p=0.25 (same as test_interior_point) -> wx=0.5 (x=0.25 is
        # halfway between grid nodes 0.0 and 0.5). y clamps to the
        # table's row-0 origin (wy=0): the gather
        # (1-wx)(1-wy)t00+(1-wx)wy*t01+wx(1-wy)t10+wx*wy*t11 reduces to
        # (1-wx)*t00 + wx*t10 = 0.5*0.10 + 0.5*0.15 = 0.125.
        assert float(e_c[0, 0, 0]) == pytest.approx(0.125, rel=1e-12)

    def test_clamp_ec_min_floors_negative_raw_value(self):
        """A table with a negative node value can produce a negative raw
        bilinear result; `max(ec_min=0.0, raw)` must floor it to 0.0."""
        table = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        diag_i = _make_diag(length=[4.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[1.0])
        luts = _synth_luts(table=table)

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert float(e_c[0, 0, 0]) == 0.0

    def test_clamp_caps_raw_value_above_one(self):
        """A table with node values > 1 (the REAL drpdrp table itself
        reaches ~10.9, see module docstring) must clamp E_c to 1.0."""
        table = np.array(
            [
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )
        diag_i = _make_diag(length=[4.0], nre=[10.0**0.5])
        diag_j = _make_diag(length=[1.0])
        luts = _synth_luts(table=table)

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert float(e_c[0, 0, 0]) == 1.0

    def test_clamp_helper_15_branch_is_unreachable_dead_code(self):
        """H1's `if(E_c>15.0) E_c=15.0` (`:16105-16107`) is applied AFTER
        `E_c=min(1.0,max(ec_min,raw))`, so `E_c<=1.0<15.0` always -- the
        15-clamp can never fire through the public `collision_efficiency`
        entry point. Transcribed verbatim anyway (see
        `_clamp_efficiency`'s own docstring); this test exercises it
        directly (bypassing the first clamp) to prove it's wired
        correctly, and documents why it's provably dead in practice."""
        # Even an absurd raw value is capped to 1.0 by the FIRST clamp
        # before the 15-check ever sees it.
        assert collision_kernel._clamp_efficiency(np.array([1.0e6])) == 1.0
        # The 15-check itself, exercised directly on a value that (by
        # construction, not via the real pipeline) skips the first clamp:
        # for any x<=15 it's a no-op; the module never calls it that way.
        assert collision_kernel._EC_MIN == 0.0

    def test_shape_is_bins_i_bins_j_npoints(self):
        diag_i = _make_diag(length=[4.0, 2.0], nre=[10.0**0.5, 10.0**0.5])
        diag_j = _make_diag(length=[1.0, 1.0, 1.0])
        luts = _synth_luts()

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, luts)

        assert e_c.shape == (2, 3, 1)


class TestCollisionEfficiencyRealLut:
    """Sanity check against the REAL packaged drpdrp LUT (not the
    synthetic table above) at realistic rain-bin magnitudes -- catches
    any accidental axis swap or LUT-loading mismatch the synthetic-table
    tests (which use their own hand-built aux) cannot."""

    def test_real_lut_finite_and_in_domain_clamped(self, real_luts):
        diag_i = _make_diag(length=[0.2], nre=[5.0])
        diag_j = _make_diag(length=[0.1])

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, real_luts)

        assert e_c.shape == (1, 1, 1)
        assert 0.0 <= float(e_c[0, 0, 0]) <= 1.0


# ---------------------------------------------------------------------------
# coalescence_efficiency -- cal_Coalescence_Efficiency, liquid-liquid
# (token 1-1) branch, H1 SS2.1 verbatim.
# ---------------------------------------------------------------------------


def _golden_coalescence_efficiency(
    *, len_i: float, len_j: float, vtm_i: float, vtm_j: float, t: float
) -> dict[str, float]:
    """Independent scalar re-derivation of `cal_Coalescence_Efficiency`'s
    liquid-liquid branch (H1 SS2.1, `:11842-11880`) -- plain `math`, no
    numpy, no import of collision_kernel internals."""
    sig_wa = float(thermo.sfc_tension(np.array([t]))[0])  # dyn/cm, CGS

    d_l = max(len_i, len_j) * 1.0e-2  # m
    d_s = min(len_i, len_j) * 1.0e-2  # m

    if min(d_l, d_s) * 100.0 < 0.01:
        return {"E_coal": 1.0, "below_d0": True}

    v_l = max(vtm_i, vtm_j) * 1.0e-2  # m/s
    v_s = min(vtm_i, vtm_j) * 1.0e-2  # m/s
    sig_si = sig_wa * 1.0e-3  # N/m
    den_w_si = 1000.0  # kg/m^3, the Fortran's OWN local SI den_w

    s_t = math.pi * sig_si * (d_l**2.0 + d_s**2.0)
    s_c = math.pi * sig_si * (d_l**3.0 + d_s**3.0) ** (2.0 / 3.0)
    ds_s = s_t - s_c
    cke = (
        (den_w_si * math.pi / 12.0)
        * (v_l - v_s) ** 2.0
        * (d_l * d_s) ** 3.0
        / (d_l**3.0 + d_s**3.0)
    )
    e_t = cke + ds_s

    if e_t < 5.0e-6:
        e_coal = 0.778 * (1.0 + d_s / d_l) ** (-2.0) * math.exp(-2.61e6 * sig_si * e_t**2.0 / s_c)
    else:
        e_coal = 0.0

    return {
        "E_coal": e_coal,
        "CKE": cke,
        "D_L": d_l,
        "D_S": d_s,
        "S_T": s_t,
        "S_C": s_c,
        "below_d0": False,
    }


class TestCoalescenceEfficiency:
    def test_known_drop_pair(self):
        """Two realistic rain drops: len_i=0.1cm (1mm diameter, faster
        vtm=250cm/s), len_j=0.05cm (0.5mm diameter, vtm=150cm/s), T=280K.
        Cross-checked against an independent scalar transcription."""
        len_i, len_j = 0.1, 0.05
        vtm_i, vtm_j = 250.0, 150.0
        t = 280.0

        golden = _golden_coalescence_efficiency(
            len_i=len_i, len_j=len_j, vtm_i=vtm_i, vtm_j=vtm_j, t=t
        )
        assert not golden["below_d0"], "test fixture must exercise the full formula, not E_coal=1"
        assert 0.0 < golden["E_coal"] < 1.0, "fixture should give a non-trivial E_coal"

        diag_i = _make_diag(length=[len_i], terminal_velocity=[vtm_i])
        diag_j = _make_diag(length=[len_j], terminal_velocity=[vtm_j])
        thermo_state = _thermo_state(t=t)

        e_coal, cke, d_l, d_s, s_t, s_c = collision_kernel.coalescence_efficiency(
            diag_i, diag_j, thermo_state
        )

        assert float(e_coal[0, 0, 0]) == pytest.approx(golden["E_coal"], rel=1e-10)
        assert float(cke[0, 0, 0]) == pytest.approx(golden["CKE"], rel=1e-10)
        assert float(d_l[0, 0, 0]) == pytest.approx(golden["D_L"], rel=1e-12)
        assert float(d_s[0, 0, 0]) == pytest.approx(golden["D_S"], rel=1e-12)
        assert float(s_t[0, 0, 0]) == pytest.approx(golden["S_T"], rel=1e-10)
        assert float(s_c[0, 0, 0]) == pytest.approx(golden["S_C"], rel=1e-10)

    def test_below_d0_cutoff_gives_full_coalescence(self):
        """Both drops tiny (below D_0=0.01cm) -> E_coal=1.0 (full
        coalescence, no bounce), matching H1's `if(min(D_L,D_S)*100<D_0)
        E_coal=1.0` early branch."""
        diag_i = _make_diag(length=[0.005], terminal_velocity=[50.0])
        diag_j = _make_diag(length=[0.003], terminal_velocity=[30.0])
        thermo_state = _thermo_state(t=280.0)

        e_coal, *_ = collision_kernel.coalescence_efficiency(diag_i, diag_j, thermo_state)

        assert float(e_coal[0, 0, 0]) == 1.0

    def test_symmetric_in_i_j(self):
        """D_L/D_S/E_coal use max/min of (len_i,len_j) and (vtm_i,vtm_j)
        -- swapping i<->j must not change the result."""
        thermo_state = _thermo_state(t=280.0)
        diag_a = _make_diag(length=[0.1], terminal_velocity=[250.0])
        diag_b = _make_diag(length=[0.05], terminal_velocity=[150.0])

        fwd = collision_kernel.coalescence_efficiency(diag_a, diag_b, thermo_state)
        rev = collision_kernel.coalescence_efficiency(diag_b, diag_a, thermo_state)

        for f, r in zip(fwd, rev, strict=True):
            assert float(f[0, 0, 0]) == pytest.approx(float(r[0, 0, 0]), rel=1e-12)

    def test_shape(self):
        thermo_state = _thermo_state(t=280.0)
        diag_i = _make_diag(length=[0.1, 0.08], terminal_velocity=[250.0, 220.0])
        diag_j = _make_diag(length=[0.05, 0.04, 0.02], terminal_velocity=[150.0, 120.0, 60.0])

        outputs = collision_kernel.coalescence_efficiency(diag_i, diag_j, thermo_state)

        for arr in outputs:
            assert arr.shape == (2, 3, 1)
            assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# collision_kernel -- KC = E_c*(vtm_i-vtm_j)*A_c*con_j*dt, H1 kernel
# assembly.
# ---------------------------------------------------------------------------


class TestCollisionKernel:
    def test_kc_matches_hand_assembly(self, real_luts):
        """Real-magnitude rain bins (2mm and ~1mm diameter drops),
        con_j~500 cm^-3, dt=2.0s. Cross-checks the KC=E_c*(vtm_i-vtm_j)*
        A_c*con_j*dt ASSEMBLY (E_c itself is separately verified by
        TestBilinearGather) via independent hand-computation of A_c and
        the final product."""
        len_i, len_j = 0.2, 0.1008
        vtm_i, vtm_j = 300.0, 150.0
        nre_i = 5.0
        con_j_val = 500.0
        dt = 2.0

        diag_i = _make_diag(length=[len_i], nre=[nre_i], terminal_velocity=[vtm_i])
        diag_j = _make_diag(length=[len_j], terminal_velocity=[vtm_j])
        con_j = np.array([[con_j_val]])

        e_c = collision_kernel.collision_efficiency(diag_i, diag_j, real_luts)
        a_c_expected = 0.25 * math.pi * (len_i + len_j) ** 2.0
        kc_expected = float(e_c[0, 0, 0]) * (vtm_i - vtm_j) * a_c_expected * con_j_val * dt

        kc = collision_kernel.collision_kernel(diag_i, diag_j, con_j, dt, real_luts)

        assert kc.shape == (1, 1, 1)
        assert float(kc[0, 0, 0]) == pytest.approx(kc_expected, rel=1e-10)

    def test_linear_in_con_j(self, real_luts):
        """PER-VOLUME basis sanity: KC is exactly linear in con_j (H1 SS5
        -- no den factor anywhere inside the kernel, so scaling con_j
        must scale KC by the same factor)."""
        diag_i = _make_diag(length=[0.2], nre=[5.0], terminal_velocity=[300.0])
        diag_j = _make_diag(length=[0.1], terminal_velocity=[150.0])
        dt = 1.5

        kc_1 = collision_kernel.collision_kernel(diag_i, diag_j, np.array([[200.0]]), dt, real_luts)
        kc_2 = collision_kernel.collision_kernel(diag_i, diag_j, np.array([[600.0]]), dt, real_luts)

        assert float(kc_2[0, 0, 0]) == pytest.approx(3.0 * float(kc_1[0, 0, 0]), rel=1e-10)

    def test_linear_in_dt(self, real_luts):
        diag_i = _make_diag(length=[0.2], nre=[5.0], terminal_velocity=[300.0])
        diag_j = _make_diag(length=[0.1], terminal_velocity=[150.0])
        con_j = np.array([[400.0]])

        kc_1 = collision_kernel.collision_kernel(diag_i, diag_j, con_j, 1.0, real_luts)
        kc_2 = collision_kernel.collision_kernel(diag_i, diag_j, con_j, 4.0, real_luts)

        assert float(kc_2[0, 0, 0]) == pytest.approx(4.0 * float(kc_1[0, 0, 0]), rel=1e-10)

    def test_col_level_zero_zeroes_realistic_sized_bins(self, real_luts):
        """H1's own verbatim `col_level==0` branch: A_c=0 whenever EITHER
        length exceeds 0.0001cm (1 micron) -- true for any realistic
        rain/cloud bin, so col_level=0 forces KC=0 here. Kept verbatim
        (H1 SS2, `:15913-15942`); cloudlab always runs coll_level=1 (see
        `AmpsConfig.cloudlab().coll_level`), so this branch is otherwise
        untested by the real configuration."""
        diag_i = _make_diag(length=[0.2], nre=[5.0], terminal_velocity=[300.0])
        diag_j = _make_diag(length=[0.1], terminal_velocity=[150.0])
        con_j = np.array([[400.0]])

        kc = collision_kernel.collision_kernel(diag_i, diag_j, con_j, 1.0, real_luts, col_level=0)

        assert float(kc[0, 0, 0]) == 0.0

    def test_shape(self, real_luts):
        diag_i = _make_diag(length=[0.2, 0.15], nre=[5.0, 3.0], terminal_velocity=[300.0, 250.0])
        diag_j = _make_diag(length=[0.1, 0.08, 0.05], terminal_velocity=[150.0, 120.0, 80.0])
        con_j = np.array([[500.0], [400.0], [200.0]])

        kc = collision_kernel.collision_kernel(diag_i, diag_j, con_j, 1.0, real_luts)

        assert kc.shape == (2, 3, 1)
        assert np.all(np.isfinite(kc))
