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


NO_DIRECTIVES_STENCIL = """\
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
    """

SINGLE_STENCIL = """\
    !$DSL DECLARE(vt=(nproma, p_patch%nlev, p_patch%nblks_e); &
    !$DSL         vn_ie=(nproma, p_patch%nlevp1, p_patch%nblks_e))

    !$DSL CREATE(something; som_field_2)

    !$DSL START(name=run_mo_velocity_advection_stencil_07; &
    !$DSL       vn_ie=p_diag&vn_ie(:, :, 1); w=p_prog%w(:,:,1); &
    !$DSL       z_v_grad_w=z_v_grad_w(:,:,:); vn_ie_abs_tol=1e-12_wp; &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
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
    !$DSL END(name=run_mo_velocity_advection_stencil_07)
    """

MULTIPLE_STENCILS = """\
    !$DSL DECLARE(vt=(nproma, p_patch%nlev, p_patch%nblks_e); &
    !$DSL         vn_ie=(nproma, p_patch%nlevp1, p_patch%nblks_e))

    !$DSL CREATE(something; som_field_2)

    !$DSL START(name=mo_nh_diffusion_stencil_06; &
    !$DSL       vn_ie=p_diag&vn_ie(:, :, 1); w=p_prog%w(:,:,1); &
    !$DSL       z_v_grad_w=z_v_grad_w(:,:,:); vn_ie_abs_tol=1e-12_wp; &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
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
    !$DSL END(name=mo_nh_diffusion_stencil_06)

    !$DSL START(name=run_mo_velocity_advection_stencil_07; &
    !$DSL       vn_ie=p_diag&vn_ie(:, :, 1); w=p_prog%w(:,:,1); &
    !$DSL       z_v_grad_w=z_v_grad_w(:,:,:); vn_ie_abs_tol=1e-12_wp; &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx)
    !$ACC PARALLEL LOOP DEFAULT(NONE) GANG VECTOR COLLAPSE(2) ASYNC(1) IF( i_am_accel_node .AND. acc_on )
    #ifdef __LOOP_EXCHANGE
        DO jc = i_startidx, i_endidx
    !DIR$ IVDEP
    #ifdef _CRAYFTN
    !DIR$ PREFERVECTOR
    #endif
        DO jk = 1, nlev
    #else
        DO jk = 1, nlev
            DO jc = i_startidx, i_endidx
    #endif
            z_nabla2_c(jc,jk,jb) =  &
              p_nh_prog%w(jc,jk,jb)                        *p_int%geofac_n2s(jc,1,jb) + &
              p_nh_prog%w(icidx(jc,jb,1),jk,icblk(jc,jb,1))*p_int%geofac_n2s(jc,2,jb) + &
              p_nh_prog%w(icidx(jc,jb,2),jk,icblk(jc,jb,2))*p_int%geofac_n2s(jc,3,jb) + &
              p_nh_prog%w(icidx(jc,jb,3),jk,icblk(jc,jb,3))*p_int%geofac_n2s(jc,4,jb)
          ENDDO
        ENDDO
        !$ACC END PARALLEL LOOP
    !$DSL END(name=run_mo_velocity_advection_stencil_07)
    """
