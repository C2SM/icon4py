# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.settings import xp

def compute_smooth_topo(
    topography: fa.CellField[ta.wpfloat],
    grid: icon_grid.IconGrid,
    num_iterations: int = 25,
) -> fa.CellKField[ta.wpfloat]:
    """
    Computes the smoothed topography needed for the SLEVE coordinate.
    """

    topography_smoothed = topography.asnumpy().copy()

    for iter in range(num_iterations):
        nabla2_topo = xp.zeros(grid.num_cells, dtype=ta.wpfloat)
        for jb in range(grid.start_blk[1], grid.num_blks[1]):
            i_startidx, i_endidx = grid.get_indices(2, jb)
            for jc in range(i_startidx, i_endidx):
                nabla2_topo[jc] = nabla2_topo[jc] + 0.125 * nabla2_topo[jc] * grid.area[jc, jb]

        topography_smoothed = topography_smoothed + nabla2_topo

    return gtx.as_field((dims.CellDim, dims.KDim), topography_smoothed)

#  SUBROUTINE compute_smooth_topo(p_patch, p_int, topo_c, niter, topo_smt_c)
#
#    TYPE(t_patch),TARGET,INTENT(INOUT) :: p_patch
#    TYPE(t_int_state), INTENT(IN) :: p_int
#
#    ! Input fields: topography on cells
#    REAL(wp), INTENT(IN) :: topo_c(:,:)
#
#    ! number of iterations
#    INTEGER,  INTENT(IN) :: niter
#
#    ! Output fields: smooth topography on cells
#    REAL(wp), INTENT(OUT) :: topo_smt_c(:,:)
#
#    INTEGER  :: jb, jc, iter
#    INTEGER  :: i_startblk, nblks_c, i_startidx, i_endidx
#    REAL(wp) :: z_topo(nproma,1,p_patch%nblks_c),nabla2_topo(nproma,1,p_patch%nblks_c)
#
#    !-------------------------------------------------------------------------
#
#    ! Initialize auxiliary fields for topography with data and nullify nabla2 field
#    z_topo(:,1,:)      = topo_c(:,:)
#    nabla2_topo(:,1,:) = 0._wp
#
#    i_startblk = p_patch%cells%start_blk(2,1)
#    nblks_c    = p_patch%nblks_c
#
#    CALL sync_patch_array(SYNC_C,p_patch,z_topo)
#
#    ! Apply nabla2-diffusion niter times to create smooth topography
#    DO iter = 1, niter
#
#      CALL nabla2_scalar(z_topo, p_patch, p_int, nabla2_topo, &
#        &                 slev=1, elev=1, rl_start=2, rl_end=min_rlcell )
#
#      DO jb = i_startblk,nblks_c
#
#        CALL get_indices_c(p_patch, jb, i_startblk, nblks_c, &
#                           i_startidx, i_endidx, 2)
#
#        DO jc = i_startidx, i_endidx
#          z_topo(jc,1,jb) = z_topo(jc,1,jb) + 0.125_wp*nabla2_topo(jc,1,jb) &
#            &                               * p_patch%cells%area(jc,jb)
#        ENDDO
#      ENDDO
#
#      CALL sync_patch_array(SYNC_C,p_patch,z_topo)
#
#    ENDDO
#
#    ! Store smooth topography on output fields
#    topo_smt_c(:,:) = z_topo(:,1,:)
#
#  END SUBROUTINE compute_smooth_topo