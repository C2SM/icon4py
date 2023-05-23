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
    !$DSL IMPORTS()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; suffix=dsl)

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; a=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL         b=nproma,p_patch%nlev,p_patch%nblks_e; type=REAL(vp))

    !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; &
    !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
    !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); &
    !$DSL       vertical_lower=1; vertical_upper=nlev; &
    !$DSL       horizontal_lower=i_startidx; horizontal_upper=i_endidx; &
    !$DSL       accpresent=True)
    !$OMP DO PRIVATE(je,jk,jb,i_startidx,i_endidx) ICON_OMP_DEFAULT_SCHEDULE
      DO jb = i_startblk,i_endblk

        CALL get_indices_e(p_patch, jb, i_startblk, i_endblk, &
                           i_startidx, i_endidx, start_bdydiff_e, grf_bdywidth_e)

    !$ACC KERNELS IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
    vn_before(:,:,:) = p_nh_prog%vn(:,:,:)
    !$ACC END KERNELS

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
    !$DSL START PROFILE(name=apply_nabla2_to_vn_in_lateral_boundary)
    !$ACC END PARALLEL LOOP
    !$DSL END PROFILE()
    !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; noprofile=True)
    """

MULTIPLE_STENCILS = """\
    !$DSL IMPORTS()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; z_rho_e=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         z_theta_v_e=nproma,p_patch%nlev,p_patch%nblks_c; z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         z_rth_pr_1=nproma,p_patch%nlev,p_patch%nblks_c; z_rth_pr_2=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         rho_ic=nproma,p_patch%nlev,p_patch%nblks_c)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_08; wgtfac_c=p_nh%metrics%wgtfac_c(:,:,1); rho=p_nh%prog(nnow)%rho(:,:,1); rho_ref_mc=p_nh%metrics%rho_ref_mc(:,:,1); &
    !$DSL               theta_v=p_nh%prog(nnow)%theta_v(:,:,1); theta_ref_mc=p_nh%metrics%theta_ref_mc(:,:,1); rho_ic=p_nh%diag%rho_ic(:,:,1); z_rth_pr_1=z_rth_pr(:,:,1,1); &
    !$DSL               z_rth_pr_2=z_rth_pr(:,:,1,2); vertical_lower=2; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)
              !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
              !$ACC LOOP GANG VECTOR TILE(32, 4)
              DO jk = 2, nlev
    !DIR$ IVDEP
                DO jc = i_startidx, i_endidx
                  ! density at interface levels for vertical flux divergence computation
                  p_nh%diag%rho_ic(jc,jk,jb) = p_nh%metrics%wgtfac_c(jc,jk,jb) *p_nh%prog(nnow)%rho(jc,jk  ,jb) + &
                                        (1._wp-p_nh%metrics%wgtfac_c(jc,jk,jb))*p_nh%prog(nnow)%rho(jc,jk-1,jb)

                  ! perturbation density and virtual potential temperature at main levels for horizontal flux divergence term
                  ! (needed in the predictor step only)
    #ifdef __SWAPDIM
                  z_rth_pr(jc,jk,jb,1) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
                  z_rth_pr(jc,jk,jb,2) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
    #else
                  z_rth_pr(1,jc,jk,jb) =  p_nh%prog(nnow)%rho(jc,jk,jb)     - p_nh%metrics%rho_ref_mc(jc,jk,jb)
                  z_rth_pr(2,jc,jk,jb) =  p_nh%prog(nnow)%theta_v(jc,jk,jb) - p_nh%metrics%theta_ref_mc(jc,jk,jb)
    #endif
    #ifdef _OPENACC
                ENDDO
              ENDDO
              !$ACC END PARALLEL
    #endif

    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_08)


    !$DSL START STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary; &
    !$DSL       z_nabla2_e=z_nabla2_e(:,:,1); area_edge=p_patch%edges%area_edge(:,1); &
    !$DSL       fac_bdydiff_v=fac_bdydiff_v; vn=p_nh_prog%vn(:,:,1); vn_abs_tol=1e-21_wp; &
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
    !$DSL END STENCIL(name=apply_nabla2_to_vn_in_lateral_boundary)

    !$DSL START STENCIL(name=calculate_nabla2_for_w; &
    !$DSL       w=p_nh_prog%w(:,:,1); geofac_n2s=p_int%geofac_n2s(:,:,1); &
    !$DSL       z_nabla2_c=z_nabla2_c(:,:,1); z_nabla2_c_abs_tol=1e-21_wp; &
    !$DSL       z_nabla2_c_rel_tol=1e-21_wp; &
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
    !$DSL ENDIF()
    !$DSL END STENCIL(name=calculate_nabla2_for_w; noendif=true)
    """

DIRECTIVES_SAMPLE = """\
!$DSL IMPORTS()

!$DSL START CREATE()

!$DSL DECLARE(vn=p_patch%vn; vn2=p_patch%vn2)

!$DSL START STENCIL(name=mo_nh_diffusion_06; vn=p_patch%vn; &
!$DSL       a=a; b=c)

!$DSL END STENCIL(name=mo_nh_diffusion_06)

!$DSL START STENCIL(name=mo_nh_diffusion_07; xn=p_patch%xn)

!$DSL END STENCIL(name=mo_nh_diffusion_07)

!$DSL UNKNOWN_DIRECTIVE()
!$DSL END CREATE()
"""

CONSECUTIVE_STENCIL = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(z_q=nproma,p_patch%nlev; z_alpha=nproma,p_patch%nlev)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_45; z_alpha=z_alpha(:,:); vertical_lower=nlevp1; &
    !$DSL               vertical_upper=nlevp1; horizontal_lower=i_startidx; horizontal_upper=i_endidx; mergecopy=true)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_45_b; z_q=z_q(:,:); vertical_lower=1; vertical_upper=1; &
    !$DSL               horizontal_lower=i_startidx; horizontal_upper=i_endidx; mergecopy=true)

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
        !$ACC LOOP GANG VECTOR
        DO jc = i_startidx, i_endidx
          z_alpha(jc,nlevp1) = 0.0_wp
          !
          ! Note: z_q is used in the tridiagonal matrix solver for w below.
          !       z_q(1) is always zero, irrespective of w(1)=0 or w(1)/=0
          !       z_q(1)=0 is equivalent to cp(slev)=c(slev)/b(slev) in mo_math_utilities:tdma_solver_vec
          z_q(jc,1) = 0._vp
        ENDDO
        !$ACC END PARALLEL
    !$DSL END PROFILE()
    !$DSL ENDIF()

    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_45; noendif=true; noprofile=true)
    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_45_b; noendif=true; noprofile=true)

    !$DSL END CREATE()
"""


FREE_FORM_STENCIL = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(z_q=nproma,p_patch%nlev; z_alpha=nproma,p_patch%nlev)

    !$DSL INSERT(some custom fields go here)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_45; z_alpha=z_alpha(:,:); vertical_lower=nlevp1; &
    !$DSL               vertical_upper=nlevp1; horizontal_lower=i_startidx; horizontal_upper=i_endidx)

        !$ACC PARALLEL IF(i_am_accel_node) DEFAULT(NONE) ASYNC(1)
        !$ACC LOOP GANG VECTOR
        DO jc = i_startidx, i_endidx
          z_alpha(jc,nlevp1) = 0.0_wp
          !
          ! Note: z_q is used in the tridiagonal matrix solver for w below.
          !       z_q(1) is always zero, irrespective of w(1)=0 or w(1)/=0
          !       z_q(1)=0 is equivalent to cp(slev)=c(slev)/b(slev) in mo_math_utilities:tdma_solver_vec
          z_q(jc,1) = 0._vp
        ENDDO
        !$ACC END PARALLEL

    !$DSL INSERT(some custom code goes here)

    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_45)

    !$DSL END CREATE()
"""
