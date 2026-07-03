# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-check of the tmx configuration against the echoed ICON namelists.

``TmxConfig.from_fortran_dict`` reads the converted *input* namelists, which
only contain the explicitly set ``aes_vdf_config`` members; all other options
rely on the dataclass defaults matching the Fortran initialization
(``vdiff_config_init``). ICON echoes the complete configuration, but as an
anonymous positional array (derived-type namelist), so this test pins the
member order of ``t_vdiff_config`` (mo_turb_vdiff_config.f90) and compares
every value the Fortran run actually used against the Python config. If it
fails, either the Fortran member order changed or one or more values drifted
from the Python defaults — either way the pinned order below and the
``TmxConfig`` defaults need to be revisited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx
from icon4py.model.common.utils import fortran_config
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from ..fixtures import load_fortran_dict


if TYPE_CHECKING:
    from icon4py.model.common.decomposition import definitions as decomposition


#: declaration order of the t_vdiff_config members (mo_turb_vdiff_config.f90);
#: the echoed aes_vdf_nml prints the derived type in exactly this order.
_VDIFF_CONFIG_MEMBER_ORDER = (
    "lsfc_mom_flux",
    "lsfc_heat_flux",
    "pr0",
    "f_tau0",
    "f_theta0",
    "f_tau_limit_fraction",
    "f_theta_limit_fraction",
    "f_tau_decay",
    "f_theta_decay",
    "ek_ep_ratio_stable",
    "ek_ep_ratio_unstable",
    "c_f",
    "c_n",
    "c_e",
    "wmc",
    "fsl",
    "fbl",
    "lmix_max",
    "z0m_min",
    "z0m_ice",
    "z0m_oce",
    "turb",
    "use_tmx",
    "solver_type",
    "energy_type",
    "dissipation_factor",
    "use_louis",
    "use_louis_land",
    "use_louis_ice",
    "louis_constant_b",
    "use_km_const",
    "km_const",
    "use_scale_turb_energy_flux",
    "scale_turb_energy_flux",
    "smag_constant",
    "turb_prandtl",
    "rturb_prandtl",
    "km_min",
    "max_turb_scale",
    "min_sfc_wind",
    "wind_g",
    "lcuda_graph_vdf",
)

#: TmxConfig members and their t_vdiff_config equivalents (here identical)
_TMX_CONFIG_MEMBERS = (
    "solver_type",
    "energy_type",
    "dissipation_factor",
    "use_louis",
    "use_louis_land",
    "use_louis_ice",
    "louis_constant_b",
    "use_km_const",
    "km_const",
    "use_scale_turb_energy_flux",
    "scale_turb_energy_flux",
    "smag_constant",
    "turb_prandtl",
    "km_min",
    "max_turb_scale",
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_tmx_config_matches_echoed_namelist(
    tmx_config: tmx.TmxConfig,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> None:
    atm_dict = load_fortran_dict(
        experiment_description, process_props, fortran_config.ATM_DICT_FNAME
    )
    flat = atm_dict["aes_vdf_nml"]["aes_vdf_config"]
    stride = len(_VDIFF_CONFIG_MEMBER_ORDER)
    assert len(flat) % stride == 0, (
        "the echoed aes_vdf_config size is not a multiple of the pinned "
        "t_vdiff_config member count: the Fortran type changed"
    )
    # first domain (the only one in the serialized experiments)
    echoed = dict(zip(_VDIFF_CONFIG_MEMBER_ORDER, flat[:stride], strict=True))
    assert echoed["use_tmx"] is True

    for name in _TMX_CONFIG_MEMBERS:
        config_value = getattr(tmx_config, name)
        if isinstance(config_value, bool):
            assert config_value is echoed[name], name
        else:
            # the echoed values go through decimal formatting; compare loosely
            assert float(config_value) == pytest.approx(echoed[name], rel=1e-12), name

    params = tmx.TmxParams(tmx_config)
    assert params.rturb_prandtl == pytest.approx(echoed["rturb_prandtl"], rel=1e-10)
