# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import lookup_tables


# ---------------------------------------------------------------------------
# Aux literals, quoted independently from
# docs/superpowers/facts/m1/lut-files.md ("F3" below) SS3/SS4, kept separate
# from lookup_tables.py's/convert_luts.py's own tables so a transcription
# bug there cannot also hide in this test's "expected" side.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def luts() -> lookup_tables.AmpsLuts:
    return lookup_tables.load_luts()


# ---------------------------------------------------------------------------
# Shapes (F3 SS4 table: com_amps.F90 common/ECTBL/FRQTBL/RMTACTBL declared
# dims; F3 SS3 for the frequency/map/lmt_mass/znorm file dims).
# ---------------------------------------------------------------------------


class TestShapes:
    def test_collision_lut_shapes(self, luts):
        assert luts.drpdrp.shape == (201, 201)
        assert luts.hexdrp.shape == (64, 71)
        assert luts.bbcdrp.shape == (64, 71)
        assert luts.coldrp.shape == (62, 71)
        assert luts.gp1drp.shape == (37, 125)
        assert luts.gp4drp.shape == (27, 125)
        assert luts.gp8drp.shape == (21, 125)

    def test_frequency_table_shapes(self, luts):
        for arr in (luts.pol_frq, luts.pla_frq, luts.col_frq, luts.ros_frq, luts.ppo_frq):
            assert arr.shape == (51, 101)

    def test_map_shapes(self, luts):
        assert luts.mtac_map_col.shape == (50, 91, 2)
        assert luts.mtac_map_pla.shape == (50, 91, 2)

    def test_lmt_mass_shapes(self, luts):
        assert luts.lmt_mass_col.shape == (50,)
        assert luts.lmt_mass_pla.shape == (50,)

    def test_znorm_shape(self, luts):
        assert luts.znorm.shape == (451, 4)


# ---------------------------------------------------------------------------
# Aux headers vs F3 literals.
# ---------------------------------------------------------------------------


class TestAuxHeaders:
    def test_drpdrp_aux(self, luts):
        # F3 SS3: "201 201 0.004 0.005 -3.9379  0.0286"
        aux = luts.adrpdrp
        assert aux.nr == 201
        assert aux.nc == 201
        assert aux.xs == pytest.approx(0.004)
        assert aux.dx == pytest.approx(0.005)
        assert aux.ys == pytest.approx(-3.9379)
        assert aux.dy == pytest.approx(0.0286)

    def test_hexdrp_aux(self, luts):
        # F3 SS3: " 64 71 -4.5 0.1 -4.3 0.1"
        aux = luts.ahexdrp
        assert (aux.nr, aux.nc) == (64, 71)
        assert aux.xs == pytest.approx(-4.5)
        assert aux.dx == pytest.approx(0.1)
        assert aux.ys == pytest.approx(-4.3)
        assert aux.dy == pytest.approx(0.1)

    def test_bbcdrp_aux(self, luts):
        aux = luts.abbcdrp
        assert (aux.nr, aux.nc) == (64, 71)
        assert aux.xs == pytest.approx(-4.5)
        assert aux.dx == pytest.approx(0.1)
        assert aux.ys == pytest.approx(-4.3)
        assert aux.dy == pytest.approx(0.1)

    def test_coldrp_aux(self, luts):
        aux = luts.acoldrp
        assert (aux.nr, aux.nc) == (62, 71)
        assert aux.xs == pytest.approx(-4.7)
        assert aux.dx == pytest.approx(0.1)
        assert aux.ys == pytest.approx(-4.8)
        assert aux.dy == pytest.approx(0.1)

    def test_gp1drp_aux(self, luts):
        aux = luts.agp1drp
        assert (aux.nr, aux.nc) == (37, 125)
        assert aux.xs == pytest.approx(0.02)
        assert aux.dx == pytest.approx(0.02)
        assert aux.ys == pytest.approx(0.3)
        assert aux.dy == pytest.approx(0.05)

    def test_gp4drp_aux(self, luts):
        aux = luts.agp4drp
        assert (aux.nr, aux.nc) == (27, 125)
        assert aux.xs == pytest.approx(0.02)
        assert aux.dx == pytest.approx(0.02)
        assert aux.ys == pytest.approx(0.7)
        assert aux.dy == pytest.approx(0.05)

    def test_gp8drp_aux(self, luts):
        aux = luts.agp8drp
        assert (aux.nr, aux.nc) == (21, 125)
        assert aux.xs == pytest.approx(0.02)
        assert aux.dx == pytest.approx(0.02)
        assert aux.ys == pytest.approx(0.9)
        assert aux.dy == pytest.approx(0.05)

    def test_frq_aux(self, luts):
        # F3 SS3: pol_frq.dat header " 51 101"
        assert luts.frq_aux.nr == 51
        assert luts.frq_aux.nc == 101

    def test_map_aux(self, luts):
        # F3 SS3: tmp_map_col.dat header "  50  91"
        assert luts.map_col_aux.nr == 50
        assert luts.map_col_aux.nc == 91
        assert luts.map_pla_aux.nr == 50
        assert luts.map_pla_aux.nc == 91

    def test_lmt_mass_aux(self, luts):
        # F3 SS3: lmt_mass_col.dat header "  50  4" (source file has 4
        # columns; only column 2 is retained as lmt_mass_col(i)).
        assert luts.lmt_mass_col_aux.nr == 50
        assert luts.lmt_mass_col_aux.nc == 4
        assert luts.lmt_mass_pla_aux.nr == 50
        assert luts.lmt_mass_pla_aux.nc == 4


# ---------------------------------------------------------------------------
# Spot value: first data line of drop_drop_Rey4.dat (F3 SS3 quotes it: "
# 0.0 0.0 0.0 0.0 0.0 0.0 ..." -- all-zero row 1 of the 201x201 table).
# ---------------------------------------------------------------------------


class TestSpotValues:
    def test_drpdrp_first_row_is_all_zero(self, luts):
        np.testing.assert_array_equal(luts.drpdrp[0, :], np.zeros(201))

    def test_frq_tables_are_normalized_probabilities(self, luts):
        # RDCETB lines 198-215 (F3 SS6): pol/pla/col clipped to >=0 then
        # renormalized so they sum to 1 at every (i, j); ros/ppo only
        # clipped to >=0 (no renormalization), so must stay in [0, ~1] but
        # need not sum to anything in particular.
        total = luts.pol_frq + luts.pla_frq + luts.col_frq
        np.testing.assert_allclose(total, np.ones_like(total), rtol=1e-10)
        assert np.all(luts.pol_frq >= 0.0)
        assert np.all(luts.pla_frq >= 0.0)
        assert np.all(luts.col_frq >= 0.0)
        assert np.all(luts.ros_frq >= 0.0)
        assert np.all(luts.ppo_frq >= 0.0)


# ---------------------------------------------------------------------------
# Computed tables: IGP knot count (F3 SS4: nok_max=23, class_Group.F90 line
# 186).
# ---------------------------------------------------------------------------


class TestIgp:
    def test_nok_igp_is_23(self, luts):
        assert luts.vigp.nok == 23
        assert lookup_tables.NOK_IGP == 23

    def test_igp_array_shapes(self, luts):
        assert luts.vigp.x.shape == (23,)
        assert luts.vigp.a.shape == (23, 4)
        assert luts.vigp.b.shape == (23, 4)

    def test_igp_knots_strictly_increasing(self, luts):
        # F3 SS5.4: x_igp = -60.0, -55.0, ..., -1.5 (deg C, monotone).
        assert np.all(np.diff(luts.vigp.x) > 0)

    def test_igp_first_last_knots(self, luts):
        assert luts.vigp.x[0] == pytest.approx(-60.0)
        assert luts.vigp.x[-1] == pytest.approx(-1.5)

    def test_igp_b_warm_half_equals_a(self, luts):
        # F3 SS5.4 line 639: b_igp(12:23,1:4) = a_igp(12:23,1:4).
        np.testing.assert_array_equal(luts.vigp.b[11:], luts.vigp.a[11:])

    def test_igp_b_cold_half_is_hardcoded(self, luts):
        # F3 SS5.4: b_igp(1:10,:) = [0,0,0,0.76]; b_igp(11,:) is the one
        # cold-half exception.
        for i in range(10):
            np.testing.assert_allclose(luts.vigp.b[i], [0.0, 0.0, 0.0, 0.76])
        np.testing.assert_allclose(luts.vigp.b[10], [0.0431, -0.1031, 0.0, 0.76])


# ---------------------------------------------------------------------------
# Breakup fragment table sizing (F3 SS5.5 formula), cross-checked against
# F3 SS4's quoted class_Cloud_Micro.F90 declaration for 80 bins:
# bu_fd(2,62400), bu_tmass(780).
# ---------------------------------------------------------------------------


class TestBreakupFragmentTables:
    def test_80_bin_sizes_match_f3_declaration(self):
        i1d_pair_max, kk_max = lookup_tables.breakup_fragment_table_sizes(nrbin=80, jmin_bk=41)
        assert i1d_pair_max == 780
        assert kk_max == 62400

    @pytest.mark.parametrize(
        "nrbin,jmin_bk",
        [(80, 41), (40, 20), (40, 1), (10, 1)],
    )
    def test_make_breakup_fragment_tables_shapes(self, nrbin, jmin_bk):
        i1d_pair_max, kk_max = lookup_tables.breakup_fragment_table_sizes(nrbin, jmin_bk)
        bu_fd, bu_tmass = lookup_tables.make_breakup_fragment_tables(nrbin, jmin_bk)
        assert bu_fd.shape == (2, kk_max)
        assert bu_tmass.shape == (i1d_pair_max,)

    def test_make_breakup_fragment_tables_zero_filled(self):
        # cal_breakfragment's pre-loop state (F3 SS5.5): bu_fd=0.0_PS;
        # bu_tmass=0.0_PS. This module deliberately does not fill them
        # further (NEEDS_CONTEXT, see lookup_tables.py module docstring).
        bu_fd, bu_tmass = lookup_tables.make_breakup_fragment_tables(nrbin=80, jmin_bk=41)
        np.testing.assert_array_equal(bu_fd, 0.0)
        np.testing.assert_array_equal(bu_tmass, 0.0)


# ---------------------------------------------------------------------------
# All arrays finite.
# ---------------------------------------------------------------------------


def _iter_arrays(obj) -> list[np.ndarray]:
    """Recursively collect every np.ndarray field from an AmpsLuts (or
    nested aux) dataclass instance."""
    arrays: list[np.ndarray] = []
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, np.ndarray):
            arrays.append(value)
        elif dataclasses.is_dataclass(value):
            arrays.extend(_iter_arrays(value))
    return arrays


class TestFinite:
    def test_all_lut_arrays_are_finite(self, luts):
        arrays = _iter_arrays(luts)
        assert len(arrays) > 0
        for arr in arrays:
            assert np.all(np.isfinite(arr)), "found non-finite value in one of the AmpsLuts arrays"

    def test_computed_tables_are_finite(self, luts):
        assert np.all(np.isfinite(luts.osm_nh42so4.y))
        assert np.all(np.isfinite(luts.osm_sodchl.y))
        assert np.all(np.isfinite(luts.snrml.y))
        assert np.all(np.isfinite(luts.isnrml.y))


# ---------------------------------------------------------------------------
# Computed-table sanity (osmotic / normal / inverse-normal), independent of
# any spot-value ground truth (see module docstring NEEDS_CONTEXT on the
# osmotic curve x-grid).
# ---------------------------------------------------------------------------


class TestComputedTableSanity:
    def test_osm_nh42so4_grid(self, luts):
        assert luts.osm_nh42so4.n == 56
        assert luts.osm_nh42so4.xs == pytest.approx(0.0)
        assert luts.osm_nh42so4.dx == pytest.approx(0.1)
        assert luts.osm_nh42so4.y.shape == (56,)

    def test_osm_sodchl_grid(self, luts):
        assert luts.osm_sodchl.n == 61
        assert luts.osm_sodchl.xs == pytest.approx(0.0)
        assert luts.osm_sodchl.dx == pytest.approx(0.1)
        assert luts.osm_sodchl.y.shape == (61,)

    def test_snrml_is_decreasing_survival_function(self, luts):
        # y_snrml(i) = 1 - Phi(x), x = 0:5:0.01 -- monotonically decreasing
        # from ~0.5 (x=0) towards 0 (x=5).
        assert luts.snrml.n == 501
        assert luts.snrml.y[0] == pytest.approx(0.5, abs=1e-6)
        assert np.all(np.diff(luts.snrml.y) <= 0.0)
        assert luts.snrml.y[-1] < 1e-6

    def test_isnrml_is_increasing_quantile_function(self, luts):
        # y_isnrml(i) = norm.ppf(prob), prob increasing from 1e-30 to 0.5
        # -- quantile increases monotonically from a large negative value
        # towards 0.
        assert luts.isnrml.n == 501
        assert np.all(np.diff(luts.isnrml.y) >= 0.0)
        assert luts.isnrml.y[-1] == pytest.approx(0.0, abs=1e-6)
        assert luts.isnrml.y[0] < -10.0


# ---------------------------------------------------------------------------
# Osmotic x-grid: read `init_osmo_par`/`osm_ammsul`/`osm_sodchl` directly in
# mod_amps_utility.F90 (~line 12923, coordinator-authorized) to settle the
# interpolation-node x-grid exactly (it is NOT uniform, unlike an earlier
# F3-only inference this supersedes -- see lookup_tables.py module
# docstring). First/last x values and spot interpolated values recomputed
# here independently from the transcribed node arrays, not copy-pasted from
# lookup_tables.py's own literals.
# ---------------------------------------------------------------------------

# Independently re-typed from mod_amps_utility.F90:12996/13062 (osm_ammsul's
# x, and its 6.0-appended osm_sodchl counterpart), kept separate from
# lookup_tables.py's own module constants per the same "expected side
# shouldn't share a transcription bug with the code side" convention as
# test_bin_grid.py.
_EXPECTED_AMMSUL_X_FIRST_ELEVEN = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_EXPECTED_AMMSUL_X_LAST_SEVEN = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]


class TestOsmoticXGrid:
    def test_ammsul_node_count_and_domain(self):
        assert len(lookup_tables._OSM_AMMSUL_X) == 23
        assert lookup_tables._OSM_AMMSUL_X[0] == pytest.approx(0.0)
        assert lookup_tables._OSM_AMMSUL_X[-1] == pytest.approx(5.5)

    def test_sodchl_node_count_and_domain(self):
        assert len(lookup_tables._OSM_SODCHL_X) == 24
        assert lookup_tables._OSM_SODCHL_X[0] == pytest.approx(0.0)
        assert lookup_tables._OSM_SODCHL_X[-1] == pytest.approx(6.0)

    def test_ammsul_x_grid_is_nonuniform_dense_then_coarse(self):
        # 0.1 step up to 1.0 (indices 0-10), then 0.2 step up to 2.0
        # (indices 10-15), then 0.5 step to the end (indices 15-22).
        np.testing.assert_allclose(
            lookup_tables._OSM_AMMSUL_X[:11], _EXPECTED_AMMSUL_X_FIRST_ELEVEN
        )
        np.testing.assert_allclose(lookup_tables._OSM_AMMSUL_X[-7:], _EXPECTED_AMMSUL_X_LAST_SEVEN)
        np.testing.assert_allclose(np.diff(lookup_tables._OSM_AMMSUL_X[:11]), 0.1)
        np.testing.assert_allclose(np.diff(lookup_tables._OSM_AMMSUL_X[10:16]), 0.2)
        np.testing.assert_allclose(np.diff(lookup_tables._OSM_AMMSUL_X[15:]), 0.5)

    def test_sodchl_x_grid_is_ammsul_grid_plus_one_node(self):
        # osm_sodchl's x array is osm_ammsul's 23-node array with a single
        # extra node (6.0) appended (mod_amps_utility.F90:13062 vs :12996).
        np.testing.assert_array_equal(lookup_tables._OSM_SODCHL_X[:23], lookup_tables._OSM_AMMSUL_X)
        assert lookup_tables._OSM_SODCHL_X[23] == pytest.approx(6.0)

    def test_ammsul_exact_at_first_and_last_node(self):
        # osm_ammsul(x(1))=y(1), osm_ammsul(x(nt))=y(nt) exactly (no
        # interpolation needed at the nodes themselves).
        assert lookup_tables._osm_ammsul(np.array([0.0]))[0] == pytest.approx(1.0)
        assert lookup_tables._osm_ammsul(np.array([5.5]))[0] == pytest.approx(0.699)

    def test_sodchl_exact_at_first_and_last_node(self):
        assert lookup_tables._osm_sodchl(np.array([0.0]))[0] == pytest.approx(1.0)
        assert lookup_tables._osm_sodchl(np.array([6.0]))[0] == pytest.approx(1.271)

    def test_ammsul_flat_extrapolation_beyond_domain(self):
        # mod_amps_utility.F90:13003-13009 (osm_ammsul): molality>=x(nt) ->
        # out=y(nt) (flat, not linear extrapolation); molality<x(1) ->
        # out=y(1).
        assert lookup_tables._osm_ammsul(np.array([10.0]))[0] == pytest.approx(0.699)
        assert lookup_tables._osm_ammsul(np.array([-1.0]))[0] == pytest.approx(1.0)

    def test_ammsul_interpolated_between_nonaligned_nodes(self):
        # x=1.1 falls strictly between nodes (1.0, 0.640) and (1.2, 0.632)
        # (the 0.2-step region) -- recomputed here from the transcribed
        # linear-interpolation formula independently of np.interp.
        x0, y0 = 1.0, 0.640
        x1, y1 = 1.2, 0.632
        expected = (y1 - y0) / (x1 - x0) * (1.1 - x0) + y0
        assert lookup_tables._osm_ammsul(np.array([1.1]))[0] == pytest.approx(expected)

    def test_osm_nh42so4_lut_first_and_last_match_node_values(self, luts):
        # The outer LUT grid (xs=0.0, dx=0.1, n=56) exactly aligns with
        # osm_ammsul's node grid at both ends (x=0.0 and x=5.5 are nodes),
        # so no interpolation is exercised there.
        assert luts.osm_nh42so4.y[0] == pytest.approx(1.0)
        assert luts.osm_nh42so4.y[-1] == pytest.approx(0.699)

    def test_osm_sodchl_lut_first_and_last_match_node_values(self, luts):
        assert luts.osm_sodchl.y[0] == pytest.approx(1.0)
        assert luts.osm_sodchl.y[-1] == pytest.approx(1.271)


# ---------------------------------------------------------------------------
# load_luts() dataclass is frozen, matching the rest of M1's dataclasses.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_amps_luts_frozen(self, luts):
        with pytest.raises(Exception):  # noqa: B017 [dataclasses.FrozenInstanceError]
            luts.znorm = None
