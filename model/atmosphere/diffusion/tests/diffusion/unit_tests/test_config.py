# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import icon4py.model.atmosphere.diffusion.config as diffusion_config
from icon4py.model.common.config import config
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import experiment


def test_diffusion_default_config(tmp_path: pathlib.Path) -> None:
    default_config = diffusion_config.init_config()
    file = tmp_path.joinpath("default.yaml")
    default_config.to_yaml(file, config.ConfigType.DEFAULT)
    reference_file = pathlib.Path(__file__).parent.joinpath("references/diffusion_default.yaml")
    assert test_utils.diff(reference_file, file)


def test_diffusion_experiment_config(
    tmp_path: pathlib.Path, experiment: definitions.Experiment
) -> None:
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
    test_utils.assert_same_except(
        overwrites[experiment.name], configuration.get(), configuration.default
    )
