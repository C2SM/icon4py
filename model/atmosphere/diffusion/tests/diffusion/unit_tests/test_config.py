from typing import Sequence

import omegaconf as oc
import pathlib
import pytest
import icon4py.model.atmosphere.diffusion.config.config as diffusion_config
def test_diffusion_config():
    structured = oc.OmegaConf.structured(diffusion_config.DiffusionConfig)

    # this will run the __post_init__ as well
    structured_init = oc.OmegaConf.structured(diffusion_config.DiffusionConfig())
    assert_default_config(structured_init)
    assert structured_init == structured
    structured_init_custom_arg = oc.OmegaConf.structured(diffusion_config.DiffusionConfig(n_substeps=2))
    assert structured_init_custom_arg.n_substeps == 2
    assert_same_except(("n_substeps",), structured_init_custom_arg, structured)

    default_file = pathlib.Path(__file__).parent.joinpath("diffusion_default.yaml")
    default_from_yaml = oc.OmegaConf.load(default_file)
    # differences Enum values and private property not present in yaml...
    assert_same_except(("diffusion_type", "shear_type", "_nudge_max_coeff"), structured, default_from_yaml.diffusion)
    # create the data class

    # TODO - interpolation
    #      - frozen data class
    #      - yml to dataclass
    x = oc.OmegaConf.to_object(structured)

    ape_file = pathlib.Path(__file__).parent.joinpath("diffusion_ape.yaml")
    ape_from_yaml = oc.OmegaConf.load(ape_file)


def assert_same_except(properties: Sequence[str], arg1, arg2):
    assert type(arg1) == type(arg2), f"{arg1} and {arg2} are not of the same type"
    temp = arg2.copy()
    for p in properties:
        assert hasattr(arg1, p), f"object of type {type(arg1)} has not attribute {p} "
        # set these attributes to the same value for comparision later on
        arg1_attr = getattr(arg1, p)
        setattr(temp, p, arg1_attr)
    assert arg1 == temp




def assert_default_config(diffusion_default: dict)->None:
    assert diffusion_default["diffusion_type"] in (diffusion_config.DiffusionType.SMAGORINSKY_4TH_ORDER, "SMAGORINSKY_4TH_ORDER")
    assert diffusion_default["apply_to_vertical_wind"]
    assert diffusion_default["apply_to_horizontal_wind"]
    assert diffusion_default["apply_to_temperature"]
    assert diffusion_default["apply_zdiffusion_t"]
    assert not diffusion_default["compute_3d_smag_coeff"]
    assert diffusion_default["ltkeshs"]
    assert diffusion_default["type_vn_diffu"] == 1
    assert diffusion_default["type_t_diffu"] == 2
    assert diffusion_default["hdiff_efdt_ratio"] == 36.0
    assert diffusion_default["hdiff_w_efdt_ratio"] == 15.0
    assert diffusion_default["thslp_zdiffu"] == 0.025
    assert diffusion_default["thhgtd_zdiffu"] == 200.0
    assert diffusion_default["shear_type"] in (diffusion_config.TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND, "VERTICAL_OF_HORIZONTAL_WIND")
    assert diffusion_default["nudging_decay_rate"] == 2.0
    assert diffusion_default["max_nudging_coefficient"] == 0.1
    assert diffusion_default["velocity_boundary_diffusion_denom"] == 200.0
    assert diffusion_default["temperature_boundary_diffusion_denom"] == 135.0
