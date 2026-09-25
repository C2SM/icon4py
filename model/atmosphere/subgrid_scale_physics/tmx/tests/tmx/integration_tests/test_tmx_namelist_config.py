# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-checks of the tmx configuration carried by an experiment's `config.yml`.

The converter (`scripts/python/fortran_config_converter.py`) locates the tmx options
positionally in the echoed `aes_vdf_nml` namelist -- an anonymous array of the
`t_vdiff_config` members in declaration order, pinned by `unnamed_index`. These tests
validate that pin against two independent sources:

- the *input* namelist carries the explicitly set members by name; a silent shift of
  the pinned positions would make the converted values disagree with the named ones.
- the members *not* set in the input namelist reach the echo through the Fortran
  initialization (`vdiff_config_init`), so for those the converted values must equal
  the `TmxConfig` dataclass defaults, which mirror it. The defaults are not
  load-bearing for the datatests (the echo carries the actually used values), but they
  must stay truthful for direct construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import f90nml
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import TmxConfig
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.testing import datatest_utils as dt_utils, definitions

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    from icon4py.model.common.decomposition import definitions as decomposition


#: the echoed namelist, the one the converter reads the pinned positions from
_NAMELIST_ATM_FNAME = "NAMELIST_ICON_output_atm"


def _read_input_namelist(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> dict:
    """Read the experiment-specific (input) namelist shipped with the archive."""
    experiment_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    candidates = sorted(
        c for c in experiment_path.glob("NAMELIST_*") if c.name != _NAMELIST_ATM_FNAME
    )
    assert len(candidates) == 1, (
        f"expected one input namelist in {experiment_path}, got {candidates}"
    )
    return f90nml.read(candidates[0]).todict()


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_tmx_config_cross_checks_input_namelist_and_defaults(
    tmx_config: TmxConfig,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> None:
    input_dict = _read_input_namelist(experiment_description, process_props)
    # first domain (the only one in the serialized experiments)
    input_members = dict(input_dict["aes_vdf_nml"]["aes_vdf_config"][0])
    assert input_members.pop("use_tmx") is True

    defaults = TmxConfig()
    checked_by_name = 0
    for field_name, _ in common_conf_opt.ConfigOption.iter_from_config_class(TmxConfig):
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
