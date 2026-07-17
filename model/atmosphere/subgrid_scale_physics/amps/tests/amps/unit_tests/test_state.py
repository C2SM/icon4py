# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.amps import state
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import index_maps


# ---------------------------------------------------------------------------
# Property-order sanity, independently against Task 4's index maps (F2) --
# catches an accidental re-ordering in state.py without relying on state.py's
# own _sorted_props helper on the "expected" side too.
# ---------------------------------------------------------------------------


class TestPropOrder:
    def test_liquid_props_are_fortran_1_based_order(self):
        names = [p.name for p in state.LiquidState.PROPS]
        assert names == ["rmt_q", "rcon_q", "rmat_q", "rmas_q"]
        assert [p.value for p in state.LiquidState.PROPS] == [1, 2, 3, 4]

    def test_ice_props_are_fortran_1_based_order(self):
        names = [p.name for p in state.IceState.PROPS]
        assert names == [
            "imt_q",
            "icon_q",
            "ivcs_q",
            "iacr_q",
            "iccr_q",
            "idcr_q",
            "iag_q",
            "icg_q",
            "inex_q",
            "imr_q",
            "imc_q",
            "imw_q",
            "imat_q",
            "imas_q",
            "ima_q",
            "imf_q",
        ]
        assert [p.value for p in state.IceState.PROPS] == list(range(1, 17))

    def test_aerosol_props_are_fortran_1_based_order(self):
        names = [p.name for p in state.AerosolState.PROPS]
        assert names == ["amt_q", "acon_q", "ams_q"]
        assert [p.value for p in state.AerosolState.PROPS] == [1, 2, 3]

    def test_props_come_from_task4_index_maps(self):
        assert set(state.LiquidState.PROPS) <= set(index_maps.LiquidPPV)
        assert set(state.IceState.PROPS) <= set(index_maps.IcePPV)
        assert set(state.AerosolState.PROPS) <= set(index_maps.AerosolPPV)


# ---------------------------------------------------------------------------
# Round trip: to_fields() -> from_fields() must be lossless (exact equality).
# ---------------------------------------------------------------------------


def _random_values(cls: type[state._BinnedState], nbins: int, ncells: int, nlev: int, seed: int):
    rng = np.random.default_rng(seed)
    npoints = ncells * nlev
    return rng.uniform(-1.0e3, 1.0e3, size=(len(cls.PROPS), nbins, 1, npoints))


class TestBinnedStateRoundTrip:
    @pytest.mark.parametrize(
        "cls", [state.LiquidState, state.IceState, state.AerosolState], ids=lambda c: c.__name__
    )
    @pytest.mark.parametrize("nbins,ncells,nlev", [(1, 3, 5), (5, 7, 11), (9, 4, 3)])
    def test_round_trip_exact(self, cls, nbins, ncells, nlev):
        values = _random_values(cls, nbins, ncells, nlev, seed=hash((cls.__name__, nbins)) % 2**32)
        bundle = cls(values=values)

        fields = bundle.to_fields(nlev)
        assert len(fields) == len(cls.PROPS) * nbins

        recovered = cls.from_fields(fields, nbins=nbins)
        assert recovered.values.shape == bundle.values.shape
        assert np.array_equal(recovered.values, bundle.values)

    def test_field_naming_convention(self):
        values = _random_values(state.LiquidState, nbins=3, ncells=2, nlev=4, seed=1)
        bundle = state.LiquidState(values=values)
        fields = bundle.to_fields(nlev=4)
        assert "liquid_rmt_q_00" in fields
        assert "liquid_rmt_q_01" in fields
        assert "liquid_rmt_q_02" in fields
        assert "liquid_rcon_q_00" in fields
        assert "liquid_rmat_q_02" in fields

    def test_ice_field_naming_convention(self):
        values = _random_values(state.IceState, nbins=1, ncells=2, nlev=3, seed=2)
        bundle = state.IceState(values=values)
        fields = bundle.to_fields(nlev=3)
        assert "ice_imas_q_00" in fields
        assert "ice_imt_q_00" in fields

    def test_to_fields_shapes(self):
        ncells, nlev = 6, 5
        values = _random_values(state.AerosolState, nbins=2, ncells=ncells, nlev=nlev, seed=3)
        bundle = state.AerosolState(values=values)
        fields = bundle.to_fields(nlev)
        for f in fields.values():
            assert f.asnumpy().shape == (ncells, nlev)


class TestThermoStateRoundTrip:
    def test_round_trip_exact(self):
        rng = np.random.default_rng(11)
        ncells, nlev = 5, 9
        npoints = ncells * nlev
        values = rng.uniform(100.0, 400.0, size=(len(state.ThermoState.PROPS), 1, 1, npoints))
        bundle = state.ThermoState(values=values)

        fields = bundle.to_fields(nlev)
        assert len(fields) == len(state.ThermoState.PROPS)
        assert "thermo_ptotv_00" in fields
        assert "thermo_momv_00" in fields

        recovered = state.ThermoState.from_fields(fields)
        assert np.array_equal(recovered.values, bundle.values)

    def test_from_fields_rejects_nbins_other_than_1(self):
        values = np.zeros((len(state.ThermoState.PROPS), 1, 1, 4))
        bundle = state.ThermoState(values=values)
        fields = bundle.to_fields(nlev=2)
        with pytest.raises(ValueError):
            state.ThermoState.from_fields(fields, nbins=2)

    def test_constructor_rejects_nbins_other_than_1(self):
        values = np.zeros((len(state.ThermoState.PROPS), 2, 1, 4))
        with pytest.raises(ValueError):
            state.ThermoState(values=values)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_base_class_not_instantiable(self):
        with pytest.raises(TypeError):
            state._BinnedState(values=np.zeros((1, 1, 1, 1)))

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            state.LiquidState(values=np.zeros((4, 2, 1)))

    def test_wrong_nprops_raises(self):
        with pytest.raises(ValueError):
            state.LiquidState(values=np.zeros((3, 2, 1, 5)))

    def test_ncat_other_than_1_raises_on_to_fields(self):
        values = np.zeros((len(state.LiquidState.PROPS), 2, 2, 10))
        bundle = state.LiquidState(values=values)
        with pytest.raises(NotImplementedError):
            bundle.to_fields(nlev=5)

    def test_to_fields_rejects_non_divisible_nlev(self):
        values = _random_values(state.LiquidState, nbins=1, ncells=3, nlev=5, seed=4)
        bundle = state.LiquidState(values=values)
        with pytest.raises(ValueError):
            bundle.to_fields(nlev=4)

    def test_from_fields_missing_key_raises(self):
        values = _random_values(state.LiquidState, nbins=2, ncells=2, nlev=3, seed=5)
        bundle = state.LiquidState(values=values)
        fields = bundle.to_fields(nlev=3)
        del fields["liquid_rmt_q_01"]
        with pytest.raises(KeyError):
            state.LiquidState.from_fields(fields, nbins=2)

    def test_from_fields_rejects_nonpositive_nbins(self):
        with pytest.raises(ValueError):
            state.LiquidState.from_fields({}, nbins=0)
