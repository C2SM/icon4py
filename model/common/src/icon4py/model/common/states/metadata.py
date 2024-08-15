

import gt4py.next as gtx

import icon4py.model.common.io.cf_utils as cf_utils
from icon4py.model.common import dimension as dims, type_alias as ta


attrs = {"functional_determinant_of_metrics_on_interface_levels":dict(
            standard_name="functional_determinant_of_metrics_on_interface_levels",
            long_name="functional determinant of the metrics [sqrt(gamma)] on half levels",
            units="",
            dims=(dims.CellDim, dims.KHalfDim),
            dtype=ta.wpfloat,
            icon_var_name="ddqz_z_half",
            ), 
        "height": dict(standard_name="height", 
                       long_name="height", 
                       units="m", 
                       dims=(dims.CellDim, dims.KDim), 
                       icon_var_name="z_mc", dtype = ta.wpfloat) ,
        "height_on_interface_levels": dict(standard_name="height_on_interface_levels", 
                                           long_name="height_on_interface_levels", 
                                           units="m", 
                                           dims=(dims.CellDim, dims.KHalfDim), 
                                           icon_var_name="z_ifc", 
                                           dtype = ta.wpfloat),
        "model_level_number": dict(standard_name="model_level_number", 
                                   long_name="model level number", 
                                   units="", dims=(dims.KDim,), 
                                   icon_var_name="k_index", 
                                   dtype = gtx.int32),
        cf_utils.INTERFACE_LEVEL_STANDARD_NAME: dict(standard_name=cf_utils.INTERFACE_LEVEL_STANDARD_NAME,
                               long_name="model interface level number",
                               units="", dims=(dims.KHalfDim,),
                               icon_var_name="k_index",
                               dtype=gtx.int32),
    }