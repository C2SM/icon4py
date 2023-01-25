from typing import Optional

import numpy as np

from icon4py.diffusion.diagnostic_state import DiagnosticState
from icon4py.diffusion.diffusion import Diffusion, DiffusionConfig, DiffusionParams
from icon4py.diffusion.icon_grid import IconGrid, VerticalModelParams
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.diffusion.prognostic_state import PrognosticState

diffusion: Diffusion(run_program=True)

def diffusion_init(vct_a: np.ndarray, nrdmax: float):
    """
    Instantiate and Initialize the diffusion object.

    should only accept simple fields as arguments for compatibility with the standalone
    Fortran ICON Diffusion component (aka Diffusion granule)

    """
    grid = IconGrid()

    vertical_params = VerticalModelParams(vct_a=vct_a, rayleigh_damping_height=nrdmax)
    config: DiffusionConfig = DiffusionConfig(grid, vertical_params)
    derived_diffusion_params = DiffusionParams(config)
    metric_state = MetricState()
    interpolation_state = InterpolationState()

    diffusion.init(config=config, params=derived_diffusion_params, vct_a=vertical_params.vct_a,)
#ELSE IF ((diffu_type == 3 .OR. diffu_type == 5) .AND. discr_vn == 1 .AND. .NOT. diffusion_config(jg)%lsmag_3d)
# mch branch
# ELSE IF ((diffu_type == 3 .OR. diffu_type == 5) .AND. discr_vn == 1) THEN t_nh_diag%vt AND t_nh_diag%theta_v_ic are used
# matching but not mch branch
# and I couldn't find discr_vn = mo_diffusion_nml:itype_vn_diffu != 1 in any experiment
def diffusion_run(dtime:float,
                  linit:bool,
                  p_int: InterpolationState,
                  vn: np.ndarray, w:np.ndarray,
                  theta_v:np.ndarray,
                  exner:np.ndarray,
                  #vt: np.ndarray(EdgeDim, KDim)
                  #theta_v_ic: np.ndarray (CellDim, KHalfDim)
                  div_ic: np.ndarray,
                  hdef_ic:np.ndarray,
                  dwdx: np.ndarray,
                  dwdy:np.ndarray,
                  ):
    diagnostic_state = DiagnosticState(hdef_ic, div_ic, dwdx, dwdy)
    prognostic_state = PrognosticState(vertical_wind=w, normal_wind=vn, exner_pressure=exner, theta_v=theta_v)
    if linit:
        diffusion.initial_step(diagnostic_state,
                               prognostic_state,
                               dtime,
                               )
    else:
        diffusion.time_step(diagnostic_state,
                               prognostic_state,
                               dtime)






class DuplicateInitializationException(Exception):
    """ Raised if the component is already initilalized"""
    pass
