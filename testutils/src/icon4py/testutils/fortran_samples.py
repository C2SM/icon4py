# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

SIMPLE_STENCIL = """\
    !#DSL STENCIL START(mo_nh_diffusion_stencil_06)
    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
    DO jk = 1, nlev
    !DIR$ IVDEP
      DO je = i_startidx, i_endidx
        p_nh_prog%vn(je,jk,jb) =   &
        p_nh_prog%vn(je,jk,jb) + &
        z_nabla2_e(je,jk,jb) * &
        p_patch%edges%area_edge(je,jb)*fac_bdydiff_v
      ENDDO
    ENDDO
    !$ACC END PARALLEL LOOP
    !#DSL STENCIL END(mo_nh_diffusion_stencil_06)
    """
