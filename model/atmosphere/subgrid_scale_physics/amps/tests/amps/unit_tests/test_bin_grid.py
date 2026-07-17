# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid


# ---------------------------------------------------------------------------
# Seed literals, quoted independently from
# docs/superpowers/facts/m1/bins-config-indexmaps.md ("F2" below) §1
# (binmicrosetup_scale) and §5 (AMPSTASK.F, the actual cloudlab config
# file). Kept separate from bin_grid.py's own module constants so that a
# transcription bug in the module cannot also hide in the test's "expected"
# side.
# ---------------------------------------------------------------------------
C_MNM = 4.188790205e-15  # F2 §1 line 63, PARAMETER c_mnm
C_MAX = 6.54498e-8  # F2 §1 line 63, PARAMETER c_max
MAXMASS_R = 5.2359870e-01  # F2 §1 line 70, PARAMETER maxmass_r
MG1 = 4.18879020478639e-12  # F2 §1 line 76, PARAMETER mg1
MG2 = 1.0e-6  # F2 §1 line 76, PARAMETER mg2
MG3 = 1.0e-2  # F2 §1 line 76, PARAMETER mg3
MG4 = 1.0e1  # F2 §1 line 77, PARAMETER mg4
NBIN40 = (8, 20, 12)  # F2 §1 line 88
NBIN20 = (4, 10, 6)  # F2 §1 line 86
NBIN10 = (2, 5, 3)  # F2 §1 line 87
# F2 §5 AMPSTASK.F line 754: minmass_s = 4.18879020478639E-12 (the actual
# cloudlab namelist value for the ice-bin seed; equals MG1 bit-for-bit).
ICE_MINMASS_CLOUDLAB = 4.18879020478639e-12


# ---------------------------------------------------------------------------
# Validation (F2 §1 lines 113-120, 258-260: PRC_abort branches -> ValueError)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_token_raises(self):
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("aerosol", 40, nbin_h=20)

    def test_liquid_invalid_nbins_raises(self):
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("liquid", 41, nbin_h=20)

    def test_liquid_valid_nbins_accepted(self):
        for nbins in (40, 80):
            grid = bin_grid.make_bin_grid("liquid", nbins, nbin_h=20)
            assert grid.nbins == nbins

    def test_ice_invalid_nbins_raises(self):
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("ice", 15)

    def test_ice_valid_nbins_accepted(self):
        for nbins in (10, 20, 40):
            grid = bin_grid.make_bin_grid("ice", nbins)
            assert grid.nbins == nbins

    def test_liquid_missing_nbin_h_raises(self):
        """nbin_h has no compile-time default in F2 (namelist-only,
        PARAM_ATMOS_PHY_MP_AMPS_bin); make_bin_grid requires it explicitly
        rather than guessing a value -- see bin_grid.py module docstring."""
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("liquid", 40)

    def test_liquid_nbin_h_too_large_raises(self):
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("liquid", 40, nbin_h=40)

    def test_liquid_nbin_h_zero_raises(self):
        with pytest.raises(ValueError):
            bin_grid.make_bin_grid("liquid", 40, nbin_h=0)


# ---------------------------------------------------------------------------
# Liquid (rain + haze) construction, F2 §1 lines 169-204.
# ---------------------------------------------------------------------------


class TestLiquidBinGrid:
    @pytest.mark.parametrize("nbins,nbin_h", [(40, 20), (80, 20), (40, 5), (40, 39)])
    def test_monotonically_increasing(self, nbins, nbin_h):
        grid = bin_grid.make_bin_grid("liquid", nbins, nbin_h=nbin_h)
        assert np.all(np.diff(grid.binb) > 0)

    @pytest.mark.parametrize("nbins,nbin_h", [(40, 20), (80, 20), (40, 5), (40, 39)])
    def test_recurrence_self_check(self, nbins, nbin_h):
        """binb[i+1] == a_b[i]*binb[i] + b_b[i] for all i, to 1e-12."""
        grid = bin_grid.make_bin_grid("liquid", nbins, nbin_h=nbin_h)
        reconstructed = grid.a_b * grid.binb[:-1] + grid.b_b
        np.testing.assert_allclose(reconstructed, grid.binb[1:], rtol=1e-12, atol=1e-12)

    def test_first_boundary_is_c_mnm(self):
        # binb[0] = c_mnm is nbin_h-independent (F2 §1 line 170).
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.binb[0] == pytest.approx(C_MNM, rel=1e-14)

    def test_last_boundary_approx_maxmass_r(self):
        """binb[-1] is LOOP-COMPUTED via dsrat, not the literal MAXMASS_R:
        F2 §1's do-loop upper bound (`nbr+1`) overwrites the earlier
        `binbr(nbr+1)=maxmass_r` direct assignment on its final iteration
        (see bin_grid.py module docstring). It is therefore only
        approximately MAXMASS_R, to floating-point rounding, not
        bit-exact."""
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.binb[-1] == pytest.approx(MAXMASS_R, rel=1e-12)

    def test_haze_boundary_is_c_max(self):
        # binb[nbin_h] = c_max is a DIRECT assignment (F2 §1 line 177), not
        # overwritten by either loop -- exact to double precision.
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.binb[20] == pytest.approx(C_MAX, rel=1e-14)

    def test_nbin_h_recorded_as_haze_split_index(self):
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.nbin_h == 20

    def test_haze_segment_ratio(self):
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        dsrat_h = (C_MAX / C_MNM) ** (1.0 / 20)
        np.testing.assert_allclose(grid.a_b[:20], dsrat_h, rtol=1e-14)

    def test_main_segment_ratio(self):
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        dsrat = (MAXMASS_R / C_MAX) ** (1.0 / (40 - 20))
        np.testing.assert_allclose(grid.a_b[20:], dsrat, rtol=1e-14)

    def test_b_b_is_zero(self):
        # No additive term (`sadd_r`) appears in the reachable binbr
        # construction (F2 §1) -- pure geometric recurrence.
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        np.testing.assert_array_equal(grid.b_b, np.zeros(40))

    def test_shapes(self):
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.binb.shape == (41,)
        assert grid.a_b.shape == (40,)
        assert grid.b_b.shape == (40,)

    def test_80_bin_config(self):
        grid = bin_grid.make_bin_grid("liquid", 80, nbin_h=20)
        assert grid.binb.shape == (81,)
        assert grid.binb[0] == pytest.approx(C_MNM, rel=1e-14)
        assert grid.binb[-1] == pytest.approx(MAXMASS_R, rel=1e-12)


# ---------------------------------------------------------------------------
# Ice construction, F2 §1 lines 216-265 (mg1 -> mg2 -> mg3 -> mg4 segments).
# ---------------------------------------------------------------------------


class TestIceBinGrid:
    @pytest.mark.parametrize("nbins", [10, 20, 40])
    def test_monotonically_increasing(self, nbins):
        grid = bin_grid.make_bin_grid("ice", nbins)
        assert np.all(np.diff(grid.binb) > 0)

    @pytest.mark.parametrize("nbins", [10, 20, 40])
    def test_recurrence_self_check(self, nbins):
        grid = bin_grid.make_bin_grid("ice", nbins)
        reconstructed = grid.a_b * grid.binb[:-1] + grid.b_b
        np.testing.assert_allclose(reconstructed, grid.binb[1:], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("nbins", [10, 20, 40])
    def test_first_boundary_is_minmass_ice_default(self, nbins):
        grid = bin_grid.make_bin_grid("ice", nbins)
        assert grid.binb[0] == pytest.approx(ICE_MINMASS_CLOUDLAB, rel=1e-14)

    @pytest.mark.parametrize("nbins", [10, 20, 40])
    def test_last_boundary_approx_mg4(self, nbins):
        """Telescopes to MG4 exactly in real arithmetic because the default
        minmass_ice == MG1 bit-for-bit (F2 §5's minmass_s literal equals F2
        §1's mg1 literal); only floating-point rounding from the segment
        multiplications separates the two."""
        grid = bin_grid.make_bin_grid("ice", nbins)
        assert grid.binb[-1] == pytest.approx(MG4, rel=1e-12)

    @pytest.mark.parametrize("nbins,n1n2n3", [(40, NBIN40), (20, NBIN20), (10, NBIN10)])
    def test_segment_boundaries_match_mg_literals(self, nbins, n1n2n3):
        n1, n2, n3 = n1n2n3
        grid = bin_grid.make_bin_grid("ice", nbins)
        assert grid.binb[n1] == pytest.approx(MG2, rel=1e-12)
        assert grid.binb[n1 + n2] == pytest.approx(MG3, rel=1e-12)
        assert grid.binb[n1 + n2 + n3] == pytest.approx(MG4, rel=1e-12)

    def test_custom_minmass_ice_overrides_default(self):
        grid = bin_grid.make_bin_grid("ice", 10, minmass_ice=1.0e-13)
        assert grid.binb[0] == pytest.approx(1.0e-13, rel=1e-14)

    def test_nbin_h_is_none_for_ice(self):
        grid = bin_grid.make_bin_grid("ice", 40)
        assert grid.nbin_h is None

    def test_b_b_is_zero(self):
        grid = bin_grid.make_bin_grid("ice", 40)
        np.testing.assert_array_equal(grid.b_b, np.zeros(40))

    def test_shapes(self):
        grid = bin_grid.make_bin_grid("ice", 20)
        assert grid.binb.shape == (21,)
        assert grid.a_b.shape == (20,)
        assert grid.b_b.shape == (20,)


# ---------------------------------------------------------------------------
# Explicit cloudlab-config checks (task brief: "40 liquid / 20 ice / 40
# ice"), first/last boundaries recomputed here from F2's quoted seed
# literals (independently of bin_grid.py's own constants).
# ---------------------------------------------------------------------------


class TestCloudlabConfigs:
    def test_40liquid_first_last_boundary(self):
        # nbin_h has no F2-quoted cloudlab value (namelist-only, no
        # compile-time default) -- see NEEDS_CONTEXT note in the report.
        # First/last boundaries do not depend on nbin_h's value, so any
        # valid nbin_h exercises the same F2-quoted literals.
        grid = bin_grid.make_bin_grid("liquid", 40, nbin_h=20)
        assert grid.binb[0] == pytest.approx(C_MNM, rel=1e-14)
        assert grid.binb[-1] == pytest.approx(MAXMASS_R, rel=1e-12)

    def test_20ice_first_last_boundary(self):
        grid = bin_grid.make_bin_grid("ice", 20)
        assert grid.binb[0] == pytest.approx(ICE_MINMASS_CLOUDLAB, rel=1e-14)
        assert grid.binb[-1] == pytest.approx(MG4, rel=1e-12)

    def test_40ice_first_last_boundary(self):
        grid = bin_grid.make_bin_grid("ice", 40)
        assert grid.binb[0] == pytest.approx(ICE_MINMASS_CLOUDLAB, rel=1e-14)
        assert grid.binb[-1] == pytest.approx(MG4, rel=1e-12)


# ---------------------------------------------------------------------------
# BinGrid dataclass is frozen (immutability convention, matches AmpsConfig
# and other M1 dataclasses).
# ---------------------------------------------------------------------------


class TestBinGridFrozen:
    def test_frozen(self):
        grid = bin_grid.make_bin_grid("ice", 10)
        with pytest.raises(Exception):  # noqa: B017 [dataclasses.FrozenInstanceError]
            grid.nbins = 20
