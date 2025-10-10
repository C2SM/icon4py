import logging

from icon4py.model.common.config import model_config
from icon4py.model.testing.fixtures.datatest import definitions


def test_read_default_config_from_himl_parser(caplog)->None:
    caplog.set_level(logging.DEBUG)
    default_config = model_config.HimlParser()


    component_list = default_config.get("model").get("components")
    assert len(component_list) == 2
    assert component_list[0].get("name") == "diffusion"
    assert component_list[1].get("name") == "solve_nonhydro"
    ## diffusion default config
    diffusion_default = default_config.get("diffusion")
    assert diffusion_default is not None
    assert_default_config(diffusion_default)


def assert_default_config(diffusion_default: dict)->None:
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


def test_override_default_values_from_himl_parser(caplog)->None:
    config = model_config.HimlParser()
    path = definitions.serialized_data_path().joinpath("mpitask1/exclaim_ape_R02B04/config")


    ## diffusion default config
    diffusion_default = config.get("diffusion", model_config.ConfigType.DEFAULT)
    assert_default_config(diffusion_default)

    config.process(path)
    diffusion_final = config.get("diffusion")
    assert diffusion_final.get("shear_type") == "VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND"
    assert diffusion_final.get("shear_type") != diffusion_default.get("shear_type")
    assert diffusion_final.get("hdiff_efdt_ratio") == 24.0
    assert diffusion_final.get("hdiff_efdt_ratio") != diffusion_default.get("hdiff_efdt_ratio")


def test_read_default_config_from_omega_parser(caplog)->None:
    caplog.set_level(logging.DEBUG)
    default_config = model_config.OmegaParser()


    component_list = default_config.get("model").get("components")
    assert len(component_list) == 2
    assert component_list[0].get("name") == "diffusion"
    assert component_list[1].get("name") == "solve_nonhydro"
    ## diffusion default config
    diffusion_default = default_config.get("diffusion")
    assert diffusion_default is not None
    assert_default_config(diffusion_default)



def test_override_default_values_from_himl_parser(caplog)->None:
    config = model_config.OmegaParser()
    path = definitions.serialized_data_path().joinpath("mpitask1/exclaim_ape_R02B04/config/diffusion.yaml")


    ## diffusion default config
    diffusion_default = config.get("diffusion", model_config.ConfigType.DEFAULT)
    assert_default_config(diffusion_default)

    config.process(path)
    diffusion_final = config.get("diffusion")
    assert diffusion_final.get("shear_type") == "VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND"
    assert diffusion_final.get("shear_type") != diffusion_default.get("shear_type")
    assert diffusion_final.get("hdiff_efdt_ratio") == 24.0
    assert diffusion_final.get("hdiff_efdt_ratio") != diffusion_default.get("hdiff_efdt_ratio")
