# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps import config
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps


# ---------------------------------------------------------------------------
# Literals quoted independently from
# docs/superpowers/facts/m1/bins-config-indexmaps.md ("F2" below), §5
# (AMPSTASK.F, the actual cloudlab task file, quoted FULL) and the checked-
# out `scale-rm/test/case/cloudlab/scripts/run.conf` (lines 144-159).
# Kept separate from config.py's own literals so that a transcription bug
# in the module cannot also hide in the test's "expected" side (same
# precedent as test_bin_grid.py).
# ---------------------------------------------------------------------------

# /AMPS_param/, F2 §5 (AMPSTASK.F). `debug` is intentionally excluded: F2
# §5 never sets it (see config.py module docstring NEEDS_CONTEXT note).
CLOUDLAB_AMPSTASK: dict[str, object] = {
    "level_comp": 7,  # F2 §5 line 649
    "debug_level": 1,  # F2 §5 line 651
    "coll_level": 1,  # F2 §5 line 860
    "out_type": 2,  # F2 §5 line 655
    "T_print_period": 3_600_000,  # F2 §5 line 660
    "output_format": "binary",  # F2 §5 line 664
    "token_c": 11,  # F2 §5 line 667
    "dtype_c": 0,  # F2 §5 line 669
    "hbreak_c": 0,  # F2 §5 line 671
    "flagp_c": 2,  # F2 §5 line 675
    "token_r": 1,  # F2 §5 line 679
    "dtype_r": 1,  # F2 §5 line 681
    "hbreak_r": 2,  # F2 §5 line 683
    "flagp_r": 2,  # F2 §5 line 685
    "token_s": 2,  # F2 §5 line 689
    "dtype_s": 1,  # F2 §5 line 691
    "hbreak_s": 2,  # F2 §5 line 693
    "flagp_s": 2,  # F2 §5 line 695
    "token_a": 3,  # F2 §5 line 699
    "dtype_a": (3, 4, 3, 3),  # F2 §5 line 701
    "flagp_a": 2,  # F2 §5 line 714 (active line; -3/1 alternates commented out)
    "ihabit_gm_random": 1,  # F2 §5 line 719
    "srat_c": 0.0,  # F2 §5 line 726
    "sadd_c": 0.0,  # F2 §5 line 728
    "minmass_c": 0.0,  # F2 §5 line 730
    "fcon_c": 25.0,  # F2 §5 line 732
    "srat_r": 2.540068909,  # F2 §5 line 737
    "sadd_r": 0.0,  # F2 §5 line 739
    "minmass_r": 4.188790e-09,  # F2 §5 line 741
    "sth_r": 1.02,  # F2 §5 line 743
    "srat_s": 4.1581061e00,  # F2 §5 line 750
    "sadd_s": 0.0,  # F2 §5 line 752
    "minmass_s": 4.18879020478639e-12,  # F2 §5 line 754
    "srat_a": 1.0e-6,  # F2 §5 line 758
    "sadd_a": 0.0,  # F2 §5 line 760
    "minmass_a": 1.0e-21,  # F2 §5 line 762
    "M_aps": (115.11, 115.11, 115.11, 115.11),  # F2 §5 line 772
    "M_api": (234.77, 234.77, 234.77, 234.77),  # F2 §5 line 775
    "den_aps": (1.79, 1.79, 1.79, 1.79),  # F2 §5 line 784
    "den_api": (5.683, 5.683, 5.683, 5.683),  # F2 §5 line 787
    "nu_aps": (2.0, 2.0, 2.0, 2.0),  # F2 §5 line 796
    "phi_aps": (0.75, 0.75, 0.75, 0.75),  # F2 §5 line 798
    "eps_ap": (1.0, 0.0, 0.05, 1.0),  # F2 §5 line 801
    "ap_lnsig": (  # F2 §5 line 812
        0.712949807856125,
        0.0,
        0.916290731874155,
        0.916290731874155,
    ),
    "ap_mean": (0.052e-4, 1.0e-4, 1.3e-4, 1.3e-4),  # F2 §5 line 819
    "ap_mean_cp": (132.0, 15.5, 132.0, 132.0),  # F2 §5 line 824
    "ap_sig_cp": (20.0, 1.4, 20.0, 20.0),  # F2 §5 line 828
    "N_ap_ini": (317.0, 0.0, 0.0, 0.0),  # F2 §5 line 832
    "APSNAME": ("NH4HSO4", "NH4HSO4", "NH4HSO4", "NH4HSO4"),  # F2 §5 line 851
    "DRCETB": (  # F2 §5 line 837
        "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/collision_data"
    ),
    "DRAPTB": (  # F2 §5 line 843
        "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/apact"
    ),
    "DRSTTB": (  # F2 §5 line 846
        "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/statpack"
    ),
    "act_type": 1,  # F2 §5 line 856
    "CCNMAX": 400.0,  # F2 §5 line 864
    "frac_dust": 0.01,  # F2 §5 line 868
    "nucleation_halflife": 0.0001,  # F2 §5 line 872
    "CRIC_RN_IMM": 0.25e-4,  # F2 §5 line 876
    "n_step_cl": 1,  # F2 §5 line 889
    "n_step_vp": 10,  # F2 §5 line 892
}

# micexfg = 1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1 (F2 §5 line 920, the
# ACTIVE line -- two other candidate lines are commented out immediately
# above it, lines 918-919). Field names per config.py's F2-literal-comment
# naming (see its module docstring for the 16/17 naming conflict).
CLOUDLAB_MICEXFG: dict[str, object] = {
    "print_flag": True,
    "rain_rain_coalescence": True,
    "ice_ice_aggregation": True,
    "ice_rain_riming": True,
    "update_surface_temperature": True,
    "vapor_deposition_liquid": True,
    "vapor_deposition_ice": True,
    "melting_shedding": True,
    "hydrodynamic_breakup_ice": False,
    "ice_nucleation_master": True,
    "hydrodynamic_breakup_rain": False,
    "autoconversion_cloud_droplet": False,
    "ice_nucleation_deposition": True,
    "ice_nucleation_contact": False,
    "ice_nucleation_hallett_mossop": False,
    "ice_nucleation_homogeneous": 0,
    "ice_nucleation_immersion": 0,
    "rain_collisional_breakup": True,
    "unused_19": 0,
    "unused_20": 1,
}

CLOUDLAB_MICEXFG_TUPLE = (1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1)

# &PARAM_ATMOS_PHY_MP_AMPS_bin, checked-out run.conf lines 144-159 (spin-up).
CLOUDLAB_BIN_NAMELIST: dict[str, object] = {
    "num_h_bins": (40, 20),  # run.conf:145
    "nbin_h": 20,  # run.conf:146
    "iadvv": 1,  # run.conf:147
    "ini_aerosol_prf": 3,  # run.conf:148
    "l_restart": False,  # run.conf:149
    "l_fix_aerosols": False,  # run.conf:150
    "l_sediment": True,  # run.conf:151
    "l_no_ice_heat": False,  # run.conf:152
    "l_fill_aerosols": False,  # run.conf:153
    "l_bin_shift": False,  # run.conf:154
    "l_axis_limit": True,  # run.conf:155
    "l_gaxis_version": 1,  # run.conf:156
    "l_aadv_version": 2,  # run.conf:157
    "l_reff_version": 2,  # run.conf:158
    "fix_aerosol_type": (False, False, False, False),  # run.conf:159
}

# &PARAM_ATMOS_PHY_MP_AMPS_bin, checked-out restart_run.conf lines 145-160
# (ice-seeding restart). Only two fields differ from CLOUDLAB_BIN_NAMELIST.
SEEDING_BIN_NAMELIST_OVERRIDES: dict[str, object] = {
    "num_h_bins": (40, 40),  # restart_run.conf:146
    "l_restart": True,  # restart_run.conf:150
}


# ---------------------------------------------------------------------------
# cloudlab() field-by-field equality vs the F2/run.conf-quoted literals.
# ---------------------------------------------------------------------------


class TestCloudlabAmpstaskFields:
    @pytest.mark.parametrize("name,expected", sorted(CLOUDLAB_AMPSTASK.items()))
    def test_field(self, name, expected):
        cfg = config.AmpsConfig.cloudlab()
        assert getattr(cfg, name) == expected


class TestCloudlabMicexfgFields:
    @pytest.mark.parametrize("name,expected", sorted(CLOUDLAB_MICEXFG.items()))
    def test_field(self, name, expected):
        cfg = config.AmpsConfig.cloudlab()
        assert getattr(cfg, name) == expected


class TestCloudlabBinNamelistFields:
    @pytest.mark.parametrize("name,expected", sorted(CLOUDLAB_BIN_NAMELIST.items()))
    def test_field(self, name, expected):
        cfg = config.AmpsConfig.cloudlab()
        assert getattr(cfg, name) == expected


class TestMicexfgArray:
    def test_cloudlab_micexfg_array_matches_amtask_literal(self):
        cfg = config.AmpsConfig.cloudlab()
        assert cfg.micexfg_array() == CLOUDLAB_MICEXFG_TUPLE

    def test_micexfg_array_length_20(self):
        cfg = config.AmpsConfig.cloudlab()
        assert len(cfg.micexfg_array()) == 20

    def test_micexfg_array_all_ints(self):
        cfg = config.AmpsConfig.cloudlab()
        assert all(isinstance(v, int) for v in cfg.micexfg_array())

    def test_default_config_micexfg_array_same_as_cloudlab(self):
        # AmpsConfig() has no genuine Fortran default for AMPSTASK.F fields
        # (see config.py module docstring), so its micexfg defaults are
        # ALSO the cloudlab F2 §5 values.
        assert config.AmpsConfig().micexfg_array() == CLOUDLAB_MICEXFG_TUPLE


# ---------------------------------------------------------------------------
# cloudlab_seeding(): identical to cloudlab() except num_h_bins/l_restart.
# ---------------------------------------------------------------------------


class TestCloudlabSeeding:
    def test_overridden_fields(self):
        cfg = config.AmpsConfig.cloudlab_seeding()
        for name, expected in SEEDING_BIN_NAMELIST_OVERRIDES.items():
            assert getattr(cfg, name) == expected

    def test_unchanged_bin_namelist_fields(self):
        cfg = config.AmpsConfig.cloudlab_seeding()
        for name, expected in CLOUDLAB_BIN_NAMELIST.items():
            if name in SEEDING_BIN_NAMELIST_OVERRIDES:
                continue
            assert getattr(cfg, name) == expected

    def test_unchanged_amptask_fields(self):
        cfg = config.AmpsConfig.cloudlab_seeding()
        for name, expected in CLOUDLAB_AMPSTASK.items():
            assert getattr(cfg, name) == expected

    def test_unchanged_micexfg(self):
        cfg = config.AmpsConfig.cloudlab_seeding()
        assert cfg.micexfg_array() == CLOUDLAB_MICEXFG_TUPLE

    def test_equals_cloudlab_with_two_fields_replaced(self):
        expected = dataclasses.replace(
            config.AmpsConfig.cloudlab(), num_h_bins=(40, 40), l_restart=True
        )
        assert config.AmpsConfig.cloudlab_seeding() == expected


# ---------------------------------------------------------------------------
# AmpsConfig() bare defaults: PARAM_ATMOS_PHY_MP_AMPS_bin fields default to
# the genuine Fortran module-level defaults, F2 §6a
# (scale_atmos_phy_mp_amps.F90 lines 121-179).
# ---------------------------------------------------------------------------


class TestDefaultConfigBinNamelistFields:
    def test_constructs_without_error(self):
        config.AmpsConfig()

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("num_h_bins", (40, 20)),  # F2 §6a lines 983-993, "coarse (default)"
            ("iadvv", 1),  # F2 §6a line 959
            ("ini_aerosol_prf", 3),  # F2 §6a line 955
            ("l_restart", False),  # F2 §6a line 948
            ("l_fix_aerosols", True),  # F2 §6a line 949 -- differs from cloudlab (False)
            ("l_sediment", True),  # F2 §6a line 950
            ("l_no_ice_heat", False),  # F2 §6a line 951
            ("l_fill_aerosols", False),  # F2 §6a line 952
            ("l_bin_shift", False),  # F2 §6a line 953
            ("l_axis_limit", True),  # F2 §6a line 954
            ("l_gaxis_version", 1),  # F2 §6a line 956
            ("l_aadv_version", 2),  # F2 §6a line 957
            ("l_reff_version", 2),  # F2 §6a line 958
            # F2 §6a line 960 -- differs from cloudlab (all False)
            ("fix_aerosol_type", (True, True, True, True)),
            ("amps_debug", False),  # F2 §6a line 946
            ("amps_ignore", False),  # F2 §6a line 947
            ("l_amps_dump", False),  # F2 §6a line 963
            ("amps_dump_dir", "."),  # F2 §6a line 964
            ("amps_dump_step_stride", 300),  # F2 §6a line 965
            ("amps_dump_is", 0),  # F2 §6a line 966
            ("amps_dump_ie", -1),  # F2 §6a line 967
            ("amps_dump_js", 0),  # F2 §6a line 968
            ("amps_dump_je", -1),  # F2 §6a line 969
        ],
    )
    def test_field(self, name, expected):
        assert getattr(config.AmpsConfig(), name) == expected

    def test_l_fix_aerosols_differs_from_cloudlab(self):
        """Fortran default is True; cloudlab's run.conf overrides to False
        (F2 §6a line 949 vs run.conf:150) -- a real, meaningful override,
        not a value that happens to coincide."""
        assert config.AmpsConfig().l_fix_aerosols is True
        assert config.AmpsConfig.cloudlab().l_fix_aerosols is False

    def test_fix_aerosol_type_differs_from_cloudlab(self):
        assert config.AmpsConfig().fix_aerosol_type == (True, True, True, True)
        assert config.AmpsConfig.cloudlab().fix_aerosol_type == (False, False, False, False)


# ---------------------------------------------------------------------------
# Frozen + validation.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_frozen(self):
        cfg = config.AmpsConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.level_comp = 1  # type: ignore[misc]


class TestValidation:
    def test_invalid_num_h_bins_liquid_raises(self):
        with pytest.raises(ValueError):
            config.AmpsConfig(num_h_bins=(41, 20))

    def test_invalid_num_h_bins_ice_raises(self):
        with pytest.raises(ValueError):
            config.AmpsConfig(num_h_bins=(40, 15))

    def test_valid_num_h_bins_combinations_accepted(self):
        for liq in (40, 80):
            for ice in (10, 20, 40):
                config.AmpsConfig(num_h_bins=(liq, ice), nbin_h=1)

    def test_nbin_h_out_of_range_raises(self):
        with pytest.raises(ValueError):
            config.AmpsConfig(num_h_bins=(40, 20), nbin_h=40)

    def test_nbin_h_zero_raises(self):
        with pytest.raises(ValueError):
            config.AmpsConfig(num_h_bins=(40, 20), nbin_h=0)

    def test_wrong_length_quad_field_raises(self):
        with pytest.raises(ValueError):
            config.AmpsConfig(dtype_a=(1, 2, 3))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# core/index_maps.py: IntEnum values match par_amps.F90 literals (F2 §2),
# .py_idx helper is 0-based, maxdims true parameters match F2 §3.
# ---------------------------------------------------------------------------


class TestIcePPV:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.IcePPV.imt_q, 1),
            (index_maps.IcePPV.icon_q, 2),
            (index_maps.IcePPV.ivcs_q, 3),
            (index_maps.IcePPV.iacr_q, 4),
            (index_maps.IcePPV.iccr_q, 5),
            (index_maps.IcePPV.idcr_q, 6),
            (index_maps.IcePPV.iag_q, 7),
            (index_maps.IcePPV.icg_q, 8),
            (index_maps.IcePPV.inex_q, 9),
            (index_maps.IcePPV.imr_q, 10),
            (index_maps.IcePPV.imc_q, 11),
            (index_maps.IcePPV.imw_q, 12),
            (index_maps.IcePPV.imat_q, 13),
            (index_maps.IcePPV.imas_q, 14),
            (index_maps.IcePPV.ima_q, 15),
            (index_maps.IcePPV.imf_q, 16),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1

    def test_16_members(self):
        # == max_nmoments_ice (F2 §6a line 974)
        assert len(list(index_maps.IcePPV)) == 16


class TestIceMassIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.IceMassIndex.imt, 1),
            (index_maps.IceMassIndex.imr, 2),
            (index_maps.IceMassIndex.ima, 3),
            (index_maps.IceMassIndex.imc, 4),
            (index_maps.IceMassIndex.imat, 5),
            (index_maps.IceMassIndex.imas, 6),
            (index_maps.IceMassIndex.imai, 7),
            (index_maps.IceMassIndex.imw, 8),
            (index_maps.IceMassIndex.imf, 9),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1


class TestIceMassRatioIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.IceMassRatioIndex.imr_m, 1),
            (index_maps.IceMassRatioIndex.ima_m, 2),
            (index_maps.IceMassRatioIndex.imc_m, 3),
            (index_maps.IceMassRatioIndex.imat_m, 4),
            (index_maps.IceMassRatioIndex.imas_m, 5),
            (index_maps.IceMassRatioIndex.imai_m, 6),
            (index_maps.IceMassRatioIndex.imw_m, 7),
            (index_maps.IceMassRatioIndex.imf_m, 8),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1


class TestIceAxisIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.IceAxisIndex.ivcs, 1),
            (index_maps.IceAxisIndex.iacr, 2),
            (index_maps.IceAxisIndex.iccr, 3),
            (index_maps.IceAxisIndex.idcr, 4),
            (index_maps.IceAxisIndex.iag, 5),
            (index_maps.IceAxisIndex.icg, 6),
            (index_maps.IceAxisIndex.inex, 7),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1

    def test_7_members(self):
        # == mxnnonmc (F2 §3 line 515)
        assert len(list(index_maps.IceAxisIndex)) == 7


class TestLiquidPPV:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.LiquidPPV.rmt_q, 1),
            (index_maps.LiquidPPV.rcon_q, 2),
            (index_maps.LiquidPPV.rmat_q, 3),
            (index_maps.LiquidPPV.rmas_q, 4),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1

    def test_4_members(self):
        # == max_nmoments_liq (F2 §6a line 974)
        assert len(list(index_maps.LiquidPPV)) == 4


class TestLiquidMassIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.LiquidMassIndex.rmt, 1),
            (index_maps.LiquidMassIndex.rmat, 2),
            (index_maps.LiquidMassIndex.rmas, 3),
            (index_maps.LiquidMassIndex.rmai, 4),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1


class TestLiquidMassRatioIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.LiquidMassRatioIndex.rmat_m, 1),
            (index_maps.LiquidMassRatioIndex.rmas_m, 2),
            (index_maps.LiquidMassRatioIndex.rmai_m, 3),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1


class TestAerosolPPV:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.AerosolPPV.amt_q, 1),
            (index_maps.AerosolPPV.acon_q, 2),
            (index_maps.AerosolPPV.ams_q, 3),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1

    def test_3_members(self):
        # == max_nmoments_aero (F2 §6a line 974)
        assert len(list(index_maps.AerosolPPV)) == 3


class TestAerosolMassIndex:
    @pytest.mark.parametrize(
        "member,value",
        [
            (index_maps.AerosolMassIndex.amt, 1),
            (index_maps.AerosolMassIndex.ams, 2),
            (index_maps.AerosolMassIndex.ami, 3),
        ],
    )
    def test_value(self, member, value):
        assert member.value == value
        assert member.py_idx == value - 1


class TestMaxDims:
    def test_values(self):
        # F2 §3 lines 511, 515, 518 (the FULL maxdims.F90 file's only
        # `integer,parameter` declarations).
        assert index_maps.MXNMASSCOMP == 8
        assert index_maps.MXNVOL == 2
        assert index_maps.MXNTEND == 12
        assert index_maps.MXNAXIS == 5
        assert index_maps.MXNNONMC == 7
        assert index_maps.MXNMASSCOMP_R == 3
