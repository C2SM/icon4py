# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from collections.abc import Sequence



import icon4py.model.common.config.reader as config
from icon4py.model.testing import test_utils


import icon4py.model.atmosphere.diffusion.config as diffusion_config


def test_diffusion_default_config(tmp_path:pathlib.Path):
    default_config = diffusion_config.init_config()
    file = tmp_path.joinpath("default.yaml")
    default_config.to_yaml(file, config.ConfigType.DEFAULT)
    assert_default_config(file)





    def foo():
        # this will run the __post_init__ as well
        structured_init = oc.OmegaConf.structured(diffusion_config.DiffusionConfig())
        assert_default_config(structured_init)
        assert structured_init == structured
        structured_init_custom_arg = oc.OmegaConf.structured(
            diffusion_config.DiffusionConfig(hdiff_w_efdt_ratio=0.55)
        )
        assert structured_init_custom_arg.hdiff_w_efdt_ratio == 0.55
        assert_same_except(("hdiff_w_efdt_ratio",), structured_init_custom_arg, structured)

        default_file = pathlib.Path(__file__).parent.joinpath("diffusion_default.yaml")
        default_from_yaml = oc.OmegaConf.load(default_file)
        # differences Enum values and private property not present in yaml...
        assert_same_except(
            ("diffusion_type", "shear_type", "_nudge_max_coeff"),
            structured,
            default_from_yaml.diffusion,
        )
        # create the data class

        # TODO (halungge):- interpolation
        #      - frozen data class
        #      - yml to dataclass

        ape_file = pathlib.Path(__file__).parent.joinpath("diffusion_ape.yaml")
        ape_from_yaml = oc.OmegaConf.load(ape_file)


def assert_same_except(properties: Sequence[str], arg1, arg2):
    assert type(arg1) is type(arg2), f"{arg1} and {arg2} are not of the same type"
    temp = arg2.copy()
    for p in properties:
        assert hasattr(arg1, p), f"object of type {type(arg1)} has not attribute {p} "
        # set these attributes to the same value for comparision later on
        arg1_attr = getattr(arg1, p)
        setattr(temp, p, arg1_attr)
    assert arg1 == temp


def assert_default_config(file: pathlib.Path) -> None:
    ref_file = pathlib.Path(__file__).parent.joinpath("references/diffusion_default.yaml")
    assert test_utils.diff(ref_file, file)


    # assert diffusion_default["diffusion_type"] in (
    #     diffusion_config.DiffusionType.SMAGORINSKY_4TH_ORDER,
    #     "SMAGORINSKY_4TH_ORDER",
    # )
    # assert diffusion_default["apply_to_vertical_wind"]
    # assert diffusion_default["apply_to_horizontal_wind"]
    # assert diffusion_default["apply_to_temperature"]
    # assert diffusion_default["apply_zdiffusion_t"]
    # assert not diffusion_default["compute_3d_smag_coeff"]
    # assert diffusion_default["ltkeshs"]
    # assert diffusion_default["type_vn_diffu"] == 1
    # assert diffusion_default["type_t_diffu"] == 2
    # assert diffusion_default["hdiff_efdt_ratio"] == 36.0
    # assert diffusion_default["hdiff_w_efdt_ratio"] == 15.0
    # assert diffusion_default["thslp_zdiffu"] == 0.025
    # assert diffusion_default["thhgtd_zdiffu"] == 200.0
    # assert diffusion_default["shear_type"] in (
    #     diffusion_config.TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
    #     "VERTICAL_OF_HORIZONTAL_WIND",
    # )
    # assert diffusion_default["nudging_decay_rate"] == 2.0
    # assert diffusion_default["max_nudging_coefficient"] == 0.1
    # assert diffusion_default["velocity_boundary_diffusion_denom"] == 200.0
    # assert diffusion_default["temperature_boundary_diffusion_denom"] == 135.0





def assert_default_config_2(diffusion_default: dict) -> None:
    assert diffusion_default.get("diffusion_type") == "SMAGORINSKY_4TH_ORDER"
    assert diffusion_default.get("apply_to_vertical_wind")
    assert diffusion_default.get("apply_to_horizontal_wind")
    assert diffusion_default.get("apply_to_temperature")
    assert diffusion_default.get("zdiffu_t")
    assert not diffusion_default.get("compute_3d_smag_coeff")
    assert diffusion_default.get("ltkeshs")
    assert diffusion_default.get("type_vn_diffu") == 1
    assert diffusion_default.get("type_t_diffu") == 2
    assert diffusion_default.get("hdiff_efdt_ratio") == 36.0
    assert diffusion_default.get("hdiff_w_efdt_ratio") == 15.0
    assert diffusion_default.get("thslp_zdiffu") == 0.025
    assert diffusion_default.get("thhgtd_zdiffu") == 200.0
    assert diffusion_default.get("shear_type") == "VERTICAL_OF_HORIZONTAL_WIND"
    assert diffusion_default.get("nudging_decay_rate") == 2.0
    assert diffusion_default.get("max_nudging_coefficient") == 0.1
    assert diffusion_default.get("velocity_boundary_diffusion_denom") == 200.0
    assert diffusion_default.get("temperature_boundary_diffusion_denom") == 135.0

