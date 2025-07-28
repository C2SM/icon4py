# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from icon4py.model.testing import cases


def build_diffusion_config(
    experiment: cases.SerializedExperiment, ndyn_substeps: int = 5
) -> "icon4py.model.atmosphere.diffusion.diffusion.DiffusionConfig":
    """Create a DiffusionConfig instance with the settings needed for the given experiment."""

    from icon4py.model.atmosphere.diffusion import diffusion

    match experiment:
        case cases.SerializedExperiment.MCH_CH_R04B09:
            config = diffusion.DiffusionConfig(
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

        case cases.SerializedExperiment.EXCLAIM_APE:
            config = diffusion.DiffusionConfig(
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

        case _:
            raise ValueError(f"Unsupported experiment: {experiment}")

    return config
