# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-checks of the tmx configuration read from the echoed ICON namelists.

``TmxConfig.from_fortran_dict`` locates the options positionally in the echoed
``aes_vdf_nml`` namelist (an anonymous array of the ``t_vdiff_config`` members
in declaration order, pinned by the ``unnamed_index`` annotations). These
tests validate that pin against two independent sources:

- the *input* namelist dict carries the explicitly set members by name; a
  silent shift of the pinned positions would make the positionally read
  values disagree with the named ones.
- the members *not* set in the input namelist reach the echo through the
  Fortran initialization (``vdiff_config_init``), so for those the echoed
  values must equal the ``TmxConfig`` dataclass defaults, which mirror it.
  The defaults are not load-bearing for the datatests (the echo carries the
  actually used values), but they must stay truthful for direct construction.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx
from icon4py.model.common.utils import fortran_config
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from ..fixtures import load_fortran_dict


if TYPE_CHECKING:
    from icon4py.model.common.decomposition import definitions as decomposition


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_tmx_config_cross_checks_input_namelist_and_defaults(
    tmx_config: tmx.TmxConfig,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> None:
    input_dict = load_fortran_dict(
        experiment_description=experiment_description,
        process_props=process_props,
        fname=fortran_config.INPUT_DICT_FNAME,
    )
    # first domain (the only one in the serialized experiments)
    input_members = input_dict["aes_vdf_nml"]["aes_vdf_config"][0]
    assert input_members.pop("use_tmx") is True

    defaults = tmx.TmxConfig()
    checked_by_name = 0
    for field_name in (f.name for f in dataclasses.fields(tmx.TmxConfig)):
        config_value = getattr(tmx_config, field_name)
        if field_name in input_members:
            # explicitly set in the input namelist: the named input value must
            # agree with the positionally read one (order-pin cross-check)
            assert config_value == input_members[field_name], field_name
            checked_by_name += 1
        else:
            # not set in the input namelist: the echoed value comes from the
            # Fortran initialization and must equal the dataclass default
            default = getattr(defaults, field_name)
            if isinstance(default, bool):
                assert config_value is default, field_name
            else:
                # the echoed values go through decimal formatting
                assert float(config_value) == pytest.approx(float(default), rel=1e-12), field_name

    # the experiment must exercise the order-pin cross-check on at least the
    # solver and energy types
    assert checked_by_name >= 2
