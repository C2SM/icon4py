# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from gt4py.next.embedded.nd_array_field import NdArrayField
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh


# TODO: the configuration code is replicated across the codebase currently. In future, the configuration should be read from an external file.


def compare_values_shallow(value1, value2, obj_name="value"):
    # Handle comparison of NdArrayField objects
    if isinstance(value1, NdArrayField) and isinstance(value2, NdArrayField):
        try:
            xp.testing.assert_equal(value1.ndarray, value2.ndarray)  # Compare arrays for equality
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Handle comparison of dictionaries
    if isinstance(value1, dict) and isinstance(value2, dict):
        if value1.keys() != value2.keys():
            return False, f"Dict keys mismatch for {obj_name}: {value1.keys()} != {value2.keys()}"
        result, error_message = compare_objects(value1, value2, obj_name)
        if not result:
            return False, error_message

        return True, None

    # Handle comparison of tuples
    if isinstance(value1, tuple) and isinstance(value2, tuple):
        if len(value1) != len(value2):
            return False, f"Tuple length mismatch for {obj_name}: {len(value1)} != {len(value2)}"
        for index, (item1, item2) in enumerate(zip(value1, value2, strict=False)):
            result, error_message = compare_values_shallow(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Handle comparison of objects with attributes (__dict__)
    if hasattr(value1, "__dict__") and hasattr(value2, "__dict__"):
        result, error_message = compare_objects(value1, value2, obj_name)
        if not result:
            return False, error_message
        return True, None

    # Check if both values are instances of numpy scalar types
    if isinstance(value1, xp.ScalarType) and isinstance(value2, xp.ScalarType):
        if value1 != value2:
            return False, f"Value mismatch for {obj_name}: {value1} != {value2}"
        return True, None

    # Handle comparison of numpy/cupy array objects
    if isinstance(value1, xp.ndarray) and isinstance(value2, xp.ndarray):
        try:
            xp.testing.assert_equal(value1, value2)  # Compare arrays for equality
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Direct comparison for other types
    if value1 != value2:
        return False, f"Value mismatch for {obj_name}: {value1} != {value2}"

    return True, None


def compare_objects(obj1, obj2, obj_name="object"):
    # Check if both objects are instances of numpy scalar types
    if isinstance(obj1, xp.ScalarType) and isinstance(obj2, xp.ScalarType):
        if obj1 != obj2:
            return False, f"Value mismatch for {obj_name}: {obj1} != {obj2}"
        return True, None

    # Check if both objects are lists
    if isinstance(obj1, list) and isinstance(obj2, list):
        # Check if lists have the same length
        if len(obj1) != len(obj2):
            return False, f"Length mismatch for {obj_name}: {len(obj1)} != {len(obj2)}"

        # Compare each element in the lists
        for index, (item1, item2) in enumerate(zip(obj1, obj2, strict=False)):
            result, error_message = compare_objects(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Check if both objects are instances of the same class
    if obj1.__class__ != obj2.__class__:
        return False, f"Class mismatch for {obj_name}: {obj1.__class__} != {obj2.__class__}"

    # Shallowly compare the attributes of both objects
    for attr, value in vars(obj1).items():
        other_value = getattr(obj2, attr, None)
        result, error_message = compare_values_shallow(value, other_value, f"{obj_name}.{attr}")
        if not result:
            return False, error_message

    return True, None


def construct_diffusion_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return r04b09_diffusion_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_diffusion_config(ndyn_substeps)


def r04b09_diffusion_config(
    ndyn_substeps,  # imported `ndyn_substeps` fixture
) -> diffusion.DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )


def exclaim_ape_diffusion_config(ndyn_substeps):
    """Create DiffusionConfig matching EXCLAIM_APE_R04B02.

    Set values to the ones used in the  EXCLAIM_APE_R04B02 experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        zdiffu_t=False,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        smagorinski_scaling_factor=0.025,
        hdiff_temp=True,
        n_substeps=ndyn_substeps,
    )


def construct_solve_nh_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return _mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return _exclaim_ape_nonhydrostatic_config(ndyn_substeps)


def _mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps):
    """Create configuration matching the mch_chR04b09_dsl experiment."""
    config = solve_nh.NonHydrostaticConfig(
        ndyn_substeps_var=ndyn_substeps,
        divdamp_order=24,
        iau_wgt_dyn=1.0,
        divdamp_fac=0.004,
        max_nudging_coeff=0.075,
    )
    return config


def _exclaim_ape_nonhydrostatic_config(ndyn_substeps):
    """Create configuration for EXCLAIM APE experiment."""
    return solve_nh.NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=24,
        ndyn_substeps_var=ndyn_substeps,
    )
