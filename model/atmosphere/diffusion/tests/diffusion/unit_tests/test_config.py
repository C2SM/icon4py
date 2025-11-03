# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import pathlib
from collections.abc import Sequence
from typing import Any

import icon4py.model.atmosphere.diffusion.config as diffusion_config
import icon4py.model.common.config.reader as config
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import experiment


def test_diffusion_default_config(tmp_path: pathlib.Path):
    default_config = diffusion_config.init_config()
    file = tmp_path.joinpath("default.yaml")
    default_config.to_yaml(file, config.ConfigType.DEFAULT)
    assert_config_file(
        pathlib.Path(__file__).parent.joinpath("references/diffusion_default.yaml"), file
    )


def test_diffusion_experiment_config(tmp_path: pathlib.Path, experiment: definitions.Experiment):
    configuration = diffusion_config.init_config()
    exp_config = definitions.construct_diffusion_config(experiment)
    configuration.update(exp_config)
    overwrites = {
        definitions.Experiments.EXCLAIM_APE.name: (
            "hdiff_efdt_ratio",
            "smagorinski_scaling_factor",
            "apply_zdiffusion_t",
            "ndyn_substeps",
        ),
        definitions.Experiments.MCH_CH_R04B09.name: (
            "hdiff_efdt_ratio",
            "smagorinski_scaling_factor",
            "max_nudging_coefficient",
            "shear_type",
            "thhgtd_zdiffu",
            "thslp_zdiffu",
            "velocity_boundary_diffusion_denominator",
            "ndyn_substeps",
        ),
    }

    file = tmp_path.joinpath(f"diffusion_{experiment.name}.yaml")
    configuration.to_yaml(file, config.ConfigType.USER)
    assert_same_except(
        overwrites[experiment.name], configuration.config_as_type, configuration.default
    )


def assert_same_except(properties: Sequence[str], arg1: Any, arg2: Any):
    assert type(arg1) is type(arg2), f"{arg1} and {arg2} are not of the same type"
    temp = copy.deepcopy(arg2)
    for p in properties:
        assert hasattr(arg1, p), f"object of type {type(arg1)} has not attribute {p} "
        # set these attributes to the same value for comparision later on
        arg1_attr = getattr(arg1, p)
        setattr(temp, p, arg1_attr)
    assert arg1 == temp


def assert_config_file(reference_file: pathlib.Path, file: pathlib.Path) -> None:
    assert test_utils.diff(reference_file, file)
