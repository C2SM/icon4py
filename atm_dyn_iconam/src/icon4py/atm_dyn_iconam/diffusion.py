from collections import namedtuple

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import minimum, broadcast

from icon4py.common.dimension import KDim

DiffusionTupleVT = namedtuple('DiffusionParamVT', 'v t')

@field_operator
def _init_diff_multfac_vn(k4: float, dyn_substeps: float):
    con = 1.0/128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))

@program
def init_diff_multfac_vn(k4:float, dyn_substeps: float, diff_multfac_vn:Field[[KDim], float]):
    _init_diff_multfac_vn(k4, dyn_substeps, out=diff_multfac_vn)



class DiffusionConfig:
    """contains necessary parameter to configure a diffusion run:
        - encapsulates namelist parameters
        - TODO [ml] derived parameters (k2 to k8) here or in Diffusion class?
    """
    ndyn_substeps = 5
    horizontal_diffusion_order = 5
    lhdiff_rcf = True # remove if always true
    hdiff_efdt_ratio = 24.0


    lateral_boundary_denominator = DiffusionTupleVT( v=200.0, t=135.0)


class Diffusion:
    """class that configures diffusion and does one diffusion step"""
    def __init__(self, config: DiffusionConfig):
        pass
        #set smag_offset, smag_limit, diff_multfac_vn (depend on configuration: horizontal_diffusion_order, lhdiff_rcf,
        # they are smag_offset and

        # MIN(1._wp/128._wp, diffusion_config(jg)%k4*REAL(ndyn_substeps,wp)/3._wp) # different for initial run!
        #diff_mult_fac_vn: Field[[KDim] =
        #smag_offset =



