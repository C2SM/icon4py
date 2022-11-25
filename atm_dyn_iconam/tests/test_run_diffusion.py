import os
import numpy as np

try:
    import serialbox as ser
except ImportError:
    os.system(
        "git clone --recursive https://github.com/GridTools/serialbox; CC=`which gcc` CXX=`which g++` pip install serialbox/src/serialbox-python"
    )
    import serialbox as ser
from functional.iterator.embedded import np_as_located_field
from atm_dyn_iconam.tests.test_diffusion import diff_multfac_vn_numpy, smag_limit_numpy, \
    enhanced_smagorinski_factor_np
from icon4py.atm_dyn_iconam.diffusion import DiffusionConfig, Diffusion, DiffusionParams
from icon4py.common.dimension import KDim

data_path = os.path.join(os.path.dirname(__file__), "ser_data")

def read_from_ser_data(path, metadata=["nproma"], fields=[]):
    rank = 0
    serializer = ser.Serializer(ser.OpenModeKind.Read, path,
                         f"reference_icon_rank{str(rank)}")
    save_points = serializer.savepoint_list()
    print(save_points)
    field_names = serializer.fieldnames()
    print(field_names)
    savepoint = serializer.savepoint["diffusion-in"].id[0].as_savepoint()
    print(type(savepoint))
    print(savepoint)
    meta_present={}
    meta_absent=[]
    for md in metadata:
        if savepoint.metainfo.has_key(md):
            meta_present[md] = savepoint.metainfo[md]
        else:
            meta_absent.append(md)

    fields_present = {}
    fields_absent = []
    for field_name in fields:
        if field_name in field_names:
            fields_present[field_name] = serializer.read(field_name, savepoint)
        else:
            fields_absent.append(field_name)
    [print(f"field  {f} not present in savepoint") for f in fields_absent]
    [print(f"metadata  {f} not present in savepoint") for f in meta_absent]

    return fields_present, meta_present


def test_diffusion_init():
    meta, fields = read_from_ser_data(data_path, metadata = ["nproma"], fields=["a_vect"])

    config = DiffusionConfig.create_with_defaults()
    additional_parameters = DiffusionParams(config)
    vct_a = np_as_located_field(KDim)(np.zeros(config.grid.get_num_k_levels()))
    #diffusion = Diffusion(config, additional_parameters, fields["vct_a"])
    diffusion = Diffusion(config, additional_parameters, vct_a)

    ## assert static local fields are initialized and correct:
    assert diffusion.smag_offset == 0.25 * additional_parameters.K4 * config.substep_as_float()
    assert diffusion.diff_multfac_w == min(1./48., additional_parameters.K4W * config.substep_as_float())

    assert np.allclose(0.0, np.asarray(diffusion.v_vert))
    assert np.allclose(0.0, np.asarray(diffusion.u_vert))
    assert np.allclose(0.0, np.asarray(diffusion.diff_multfac_n2w))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_ec))
    assert np.allclose(0.0, np.asarray(diffusion.kh_smag_e))

    shape_k = np.asarray(diffusion.diff_multfac_vn.shape)
    expected_smag_limit = smag_limit_numpy(shape_k, additional_parameters.K4, config.substep_as_float())
    assert np.allclose(expected_smag_limit,  np.asarray(diffusion.smag_limit))

    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape_k, additional_parameters.K4, config.substep_as_float())
    assert np.allclose(expected_diff_multfac_vn, np.asarray(diffusion.diff_multfac_vn))
    expected_enh_smag_fac = enhanced_smagorinski_factor_np(additional_parameters.smagorinski_factor, additional_parameters.smagorinski_height, vct_a)
    assert np.allclose(expected_enh_smag_fac, np.asarray(diffusion.enh_smag_fac)[:-1])








