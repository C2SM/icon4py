# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data, tmx_states


def test_inputs_cover_input_and_surface_flux_states():
    state_keys = {f.name for f in dataclasses.fields(tmx_states.TmxInputState)} | {
        f.name for f in dataclasses.fields(tmx_states.TmxSurfaceFluxState)
    }
    assert set(tmx_data.INPUTS_PROPERTIES) == state_keys


def test_outputs_cover_tendencies_and_diagnostics():
    tendency_keys = {f.name for f in dataclasses.fields(tmx_states.TmxTendencyState)}
    diagnostic_keys = {
        "km",
        "kh",
        "heating",
        "dissip_ke",
        "cptgz_vi",
        "dissip_ke_vi",
        "int_energy_vi",
        "int_energy_vi_tend",
    }
    assert set(tmx_data.OUTPUTS_PROPERTIES) == tendency_keys | diagnostic_keys
    for key in tendency_keys:
        assert "tendency" in tmx_data.OUTPUTS_PROPERTIES[key]["standard_name"] or (
            tmx_data.OUTPUTS_PROPERTIES[key].get("long_name", "").startswith("tendency of")
        )
