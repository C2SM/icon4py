# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import icon4py.model.atmosphere.dycore.config as dycore_config
from icon4py.model.common.config import config
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import experiment


def test_non_hydrostatic_default_config(tmp_path: pathlib.Path) -> None:
    default_config = dycore_config.init_config()
    file = tmp_path.joinpath("default.yaml")
    default_config.to_yaml(file, config.ConfigType.DEFAULT)

    reference_file = pathlib.Path(__file__).parent.joinpath("references/dycore_default.yaml")
    assert test_utils.diff(reference_file, file)


def test_dycore_experiment_config(
    tmp_path: pathlib.Path, experiment: definitions.Experiment
) -> None:
    configuration = dycore_config.init_config()
    exp_config = definitions.construct_nonhydrostatic_config(experiment)

    configuration.update(exp_config)
    overwrites = {
        definitions.Experiments.EXCLAIM_APE.name: (
            "divdamp_order",
            "rayleigh_coeff",
            "ndyn_substeps",
        ),
        definitions.Experiments.MCH_CH_R04B09.name: (
            "divdamp_order",
            "iau_wgt_dyn",
            "fourth_order_divdamp_factor",
            "max_nudging_coefficient",
            "ndyn_substeps",
        ),
    }

    file = tmp_path.joinpath(f"dycore_{experiment.name}.yaml")
    configuration.to_yaml(file, config.ConfigType.USER)
    reference_file = definitions.config_reference_path().joinpath(f"dycore_{experiment.name}.yaml")
    assert test_utils.diff(reference_file, file)
    test_utils.assert_same_except(
        overwrites[experiment.name], configuration.get(), configuration.default
    )
