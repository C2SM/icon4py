# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


# TODO: revise names with domain scientists

Z_MC: Final[str] = "height"
DDQZ_Z_HALF: Final[str] = "functional_determinant_of_metrics_on_interface_levels"
DDQZ_Z_FULL: Final[str] = "ddqz_z_full"
INV_DDQZ_Z_FULL: Final[str] = "inv_ddqz_z_full"
SCALFAC_DD3D: Final[str] = "scaling_factor_for_3d_divergence_damping"
RAYLEIGH_W: Final[str] = "rayleigh_w"
COEFF1_DWDZ: Final[str] = "coeff1_dwdz"
COEFF2_DWDZ: Final[str] = "coeff2_dwdz"
EXNER_REF_MC: Final[str] = "exner_ref_mc"
THETA_REF_MC: Final[str] = "theta_ref_mc"
D2DEXDZ2_FAC1_MC: Final[str] = "d2dexdz2_fac1_mc"
D2DEXDZ2_FAC2_MC: Final[str] = "d2dexdz2_fac2_mc"
VERT_OUT: Final[str] = "vert_out"
DDXT_Z_HALF_E: Final[str] = "ddxt_z_half_e"
DDXN_Z_HALF_E: Final[str] = "ddxn_z_half_e"
DDXN_Z_FULL: Final[str] = "ddxn_z_full"
VWIND_IMPL_WGT: Final[str] = "vwind_impl_wgt"
VWIND_EXPL_WGT: Final[str] = "vwind_expl_wgt"
EXNER_EXFAC: Final[str] = "exner_exfac"
WGTFAC_C: Final[str] = "wgtfac_c"
WGTFAC_E: Final[str] = "wgtfac_e"
FLAT_IDX_MAX: Final[str] = "flat_idx_max"
PG_EDGEIDX_DSL: Final[str] = "edge_mask_for_pressure_gradient_extrapolation"
PG_EDGEDIST_DSL: Final[str] = "distance_for_pressure_gradient_extrapolation"
MASK_PROG_HALO_C: Final[str] = "mask_prog_halo_c"
BDY_HALO_C: Final[str] = "bdy_halo_c"
HMASK_DD3D: Final[str] = "hmask_dd3d"
ZDIFF_GRADP: Final[str] = "zdiff_gradp"
COEFF_GRADEKIN: Final[str] = "coeff_gradekin"
WGTFACQ_C: Final[str] = "weighting_factor_for_quadratic_interpolation_to_cell_surface"
WGTFACQ_E: Final[str] = "weighting_factor_for_quadratic_interpolation_to_edge_center"
MAXSLP: Final[str] = "maxslp"
MAXHGTD: Final[str] = "maxhgtd"
MAXSLP_AVG: Final[str] = "maxslp_avg"
MAXHGTD_AVG: Final[str] = "maxhgtd_avg"
MAX_NBHGT: Final[str] = "max_nbhgt"
MASK_HDIFF: Final[str] = "mask_hdiff"
ZD_DIFFCOEF_DSL: Final[str] = "zd_diffcoef_dsl"
ZD_INTCOEF_DSL: Final[str] = "zd_intcoef_dsl"
ZD_VERTOFFSET_DSL: Final[str] = "zd_vertoffset_dsl"


attrs: dict[str, model.FieldMetaData] = {
    Z_MC: dict(
        standard_name=Z_MC,
        long_name="height",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_mc",
        dtype=ta.wpfloat,
    ),
    DDQZ_Z_HALF: dict(
        standard_name=DDQZ_Z_HALF,
        long_name="functional_determinant_of_metrics_on_interface_levels",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="ddqz_z_half",
        dtype=ta.wpfloat,
    ),
    DDQZ_Z_FULL: dict(
        standard_name=DDQZ_Z_FULL,
        long_name="ddqz_z_full",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="ddqz_z_full",
        dtype=ta.wpfloat,
    ),
    INV_DDQZ_Z_FULL: dict(
        standard_name=INV_DDQZ_Z_FULL,
        long_name="inv_ddqz_z_full",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="inv_ddqz_z_full",
        dtype=ta.wpfloat,
    ),
    SCALFAC_DD3D: dict(
        standard_name=SCALFAC_DD3D,
        long_name="Scaling factor for 3D divergence damping terms",
        units="",
        dims=(dims.KDim,),
        icon_var_name="scalfac_dd3d",
        dtype=ta.wpfloat,
    ),
    RAYLEIGH_W: dict(
        standard_name=RAYLEIGH_W,
        long_name="rayleigh_w",
        units="",
        dims=(dims.KHalfDim,),
        icon_var_name="rayleigh_w",
        dtype=ta.wpfloat,
    ),
    COEFF1_DWDZ: dict(
        standard_name=COEFF1_DWDZ,
        long_name="coeff1_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="coeff1_dwdz",
        dtype=ta.wpfloat,
    ),
    COEFF2_DWDZ: dict(
        standard_name=COEFF2_DWDZ,
        long_name="coeff2_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="coeff2_dwdz",
        dtype=ta.wpfloat,
    ),
    EXNER_REF_MC: dict(
        standard_name=EXNER_REF_MC,
        long_name="exner_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="exner_ref_mc",
        dtype=ta.wpfloat,
    ),
    THETA_REF_MC: dict(
        standard_name=THETA_REF_MC,
        long_name="theta_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="theta_ref_mc",
        dtype=ta.wpfloat,
    ),
    D2DEXDZ2_FAC1_MC: dict(
        standard_name=D2DEXDZ2_FAC1_MC,
        long_name="d2dexdz2_fac1_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="d2dexdz2_fac1_mc",
        dtype=ta.wpfloat,
    ),
    D2DEXDZ2_FAC2_MC: dict(
        standard_name=D2DEXDZ2_FAC2_MC,
        long_name="d2dexdz2_fac2_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="d2dexdz2_fac2_mc",
        dtype=ta.wpfloat,
    ),
    VERT_OUT: dict(
        standard_name=VERT_OUT,
        long_name="vert_out",
        units="",
        dims=(dims.VertexDim, dims.KHalfDim),
        icon_var_name="vert_out",
        dtype=ta.wpfloat,
    ),
    DDXT_Z_HALF_E: dict(
        standard_name=DDXT_Z_HALF_E,
        long_name="ddxt_z_half_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="ddxt_z_half_e",
        dtype=ta.wpfloat,
    ),
    DDXN_Z_HALF_E: dict(
        standard_name=DDXN_Z_HALF_E,
        long_name="ddxn_z_half_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="ddxn_z_half_e",
        dtype=ta.wpfloat,
    ),
    DDXN_Z_FULL: dict(
        standard_name=DDXN_Z_FULL,
        long_name="ddxn_z_full",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="ddxn_z_full",
        dtype=ta.wpfloat,
    ),
    VWIND_IMPL_WGT: dict(
        standard_name=VWIND_IMPL_WGT,
        long_name="vwind_impl_wgt",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="vwind_impl_wgt",
        dtype=ta.wpfloat,
    ),
    VWIND_EXPL_WGT: dict(
        standard_name=VWIND_EXPL_WGT,
        long_name="vwind_expl_wgt",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="vwind_expl_wgt",
        dtype=ta.wpfloat,
    ),
    EXNER_EXFAC: dict(
        standard_name=EXNER_EXFAC,
        long_name="exner_exfac",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="exner_exfac",
        dtype=ta.wpfloat,
    ),
    WGTFAC_C: dict(
        standard_name=WGTFAC_C,
        long_name="wgtfac_c",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="wgtfac_c",
        dtype=ta.wpfloat,
    ),
    WGTFAC_E: dict(
        standard_name=WGTFAC_E,
        long_name="wgtfac_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="wgtfac_e",
        dtype=ta.wpfloat,
    ),
    FLAT_IDX_MAX: dict(
        standard_name=FLAT_IDX_MAX,
        long_name="flat_idx_max",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="flat_idx_max",
        dtype=ta.wpfloat,
    ),
    PG_EDGEIDX_DSL: dict(
        standard_name=PG_EDGEIDX_DSL,
        long_name="edge mask for pressure gradient downward extrapolation",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="pg_edgeidx_dsl",
        dtype=bool,
    ),
    PG_EDGEDIST_DSL: dict(
        standard_name=PG_EDGEDIST_DSL,
        long_name="extrapolation distance for pressure gradient downward extrapolation",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="pg_exdist_dsl",
        dtype=ta.wpfloat,
    ),
    MASK_PROG_HALO_C: dict(
        standard_name=MASK_PROG_HALO_C,
        long_name="mask_prog_halo_c",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="mask_prog_halo_c",
        dtype=bool,
    ),
    BDY_HALO_C: dict(
        standard_name=BDY_HALO_C,
        long_name="bdy_halo_c",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="bdy_halo_c",
        dtype=bool,
    ),
    HMASK_DD3D: dict(
        standard_name=HMASK_DD3D,
        long_name="hmask_dd3d",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="hmask_dd3d",
        dtype=ta.wpfloat,
    ),
    ZDIFF_GRADP: dict(
        standard_name=ZDIFF_GRADP,
        long_name="zdiff_gradp",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="zdiff_gradp",
        dtype=ta.wpfloat,
    ),
    COEFF_GRADEKIN: dict(
        standard_name=COEFF_GRADEKIN,
        long_name="coeff_gradekin",
        units="",
        dims=(dims.ECDim,),
        icon_var_name="coeff_gradekin",
        dtype=ta.wpfloat,
    ),
    WGTFACQ_C: dict(
        standard_name=WGTFACQ_C,
        long_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        dtype=ta.wpfloat,
    ),
    WGTFACQ_E: dict(
        standard_name=WGTFACQ_E,
        long_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        dtype=ta.wpfloat,
    ),
    MAXSLP: dict(
        standard_name=MAXSLP,
        long_name="maxslp",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="maxslp",
        dtype=ta.wpfloat,
    ),
    MAXHGTD: dict(
        standard_name=MAXHGTD,
        long_name="maxhgtd",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="maxhgtd",
        dtype=ta.wpfloat,
    ),
    MAXSLP_AVG: dict(
        standard_name=MAXSLP_AVG,
        long_name="maxslp_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="maxslp_avg",
        dtype=ta.wpfloat,
    ),
    MAXHGTD_AVG: dict(
        standard_name=MAXHGTD_AVG,
        long_name="maxhgtd_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="maxhgtd_avg",
        dtype=ta.wpfloat,
    ),
    MAX_NBHGT: dict(
        standard_name=MAX_NBHGT,
        long_name="max_nbhgt",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="max_nbhgt",
        dtype=ta.wpfloat,
    ),
    MASK_HDIFF: dict(
        standard_name=MASK_HDIFF,
        long_name="mask_hdiff",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="mask_hdiff",
        dtype=ta.wpfloat,
    ),
    ZD_DIFFCOEF_DSL: dict(
        standard_name=ZD_DIFFCOEF_DSL,
        long_name="zd_diffcoef_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="zd_diffcoef_dsl",
        dtype=ta.wpfloat,
    ),
    ZD_INTCOEF_DSL: dict(
        standard_name=ZD_INTCOEF_DSL,
        long_name="zd_intcoef_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="zd_intcoef_dsl",
        dtype=ta.wpfloat,
    ),
    ZD_VERTOFFSET_DSL: dict(
        standard_name=ZD_VERTOFFSET_DSL,
        long_name="zd_vertoffset_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="zd_vertoffset_dsl",
        dtype=ta.wpfloat,
    ),
}
CELL_HEIGHT_ON_INTERFACE_LEVEL = "height_on_interface_levels"
