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

    !$DSL START CREATE()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; a=nproma,p_patch%nlev,p_patch%nblks_e; &
    !$DSL         b=nproma,p_patch%nlev,p_patch%nblks_e; kind=vp)

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

    !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on ) DEFAULT(NONE) ASYNC(1)
    vn_before(:,:,:) = p_nh_prog%vn(:,:,:)
    !$ACC END PARALLEL

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
    !$DSL END CREATE()
    """

MULTIPLE_STENCILS = """\
    !$DSL IMPORTS()

    !$DSL START CREATE()

    !$DSL DECLARE(vn=nproma,p_patch%nlev,p_patch%nblks_e; z_rho_e=nproma,p_patch%nlev,p_patch%nblks_c; &
    !$DSL         z_theta_v_e=nproma,p_patch%nlev,p_patch%nblks_c; z_nabla2_c=nproma,p_patch%nlev,p_patch%nblks_c)

    !$DSL START STENCIL(name=mo_solve_nonhydro_stencil_16; p_vn=p_nh%prog(nnow)%vn(:,:,1); rho_ref_me=p_nh%metrics%rho_ref_me(:,:,1); &
    !$DSL          theta_ref_me=p_nh%metrics%theta_ref_me(:,:,1); p_distv_bary_1=p_distv_bary(:,:,1,1); p_distv_bary_2=p_distv_bary(:,:,1,2); &
    !$DSL          z_grad_rth_1=z_grad_rth(:,:,1,1); z_grad_rth_2=z_grad_rth(:,:,1,2); z_grad_rth_3=z_grad_rth(:,:,1,3); z_grad_rth_4=z_grad_rth(:,:,1,4); &
    !$DSL          z_rth_pr_1=z_rth_pr(:,:,1,1); z_rth_pr_2=z_rth_pr(:,:,1,2); z_rho_e=z_rho_e(:,:,1); z_theta_v_e=z_theta_v_e(:,:,1); &
    !$DSL          vertical_lower=1; vertical_upper=nlev; horizontal_lower=i_startidx; horizontal_upper=i_endidx)
    !$ACC PARALLEL IF( i_am_accel_node .AND. acc_on )  DEFAULT(NONE) ASYNC(1)
    #if defined (__LOOP_EXCHANGE) || defined (__SX__)
                  ! For cache-based machines, also the back-trajectory computation is inlined to improve efficiency
                  !$ACC LOOP GANG VECTOR COLLAPSE(2)   &
                  !$ACC      PRIVATE(lvn_pos,ilc0,ibc0,z_ntdistv_bary_1,z_ntdistv_bary_2,distv_bary_1,distv_bary_2)
    #ifdef __LOOP_EXCHANGE
                  DO je = i_startidx, i_endidx
    !DIR$ IVDEP, PREFERVECTOR
                    DO jk = 1, nlev
    #else
                  DO jk = 1, nlev
                    DO je = i_startidx, i_endidx
    #endif
                      lvn_pos = p_nh%prog(nnow)%vn(je,jk,jb) >= 0._wp

                      ! line and block indices of upwind neighbor cell
                      ilc0 = MERGE(p_patch%edges%cell_idx(je,jb,1),p_patch%edges%cell_idx(je,jb,2),lvn_pos)
                      ibc0 = MERGE(p_patch%edges%cell_blk(je,jb,1),p_patch%edges%cell_blk(je,jb,2),lvn_pos)

                      ! distances from upwind mass point to the end point of the backward trajectory
                      ! in edge-normal and tangential directions
                      z_ntdistv_bary_1 =  - ( p_nh%prog(nnow)%vn(je,jk,jb) * dthalf +    &
                        MERGE(p_int%pos_on_tplane_e(je,jb,1,1), p_int%pos_on_tplane_e(je,jb,2,1),lvn_pos))

                      z_ntdistv_bary_2 =  - ( p_nh%diag%vt(je,jk,jb) * dthalf +    &
                        MERGE(p_int%pos_on_tplane_e(je,jb,1,2), p_int%pos_on_tplane_e(je,jb,2,2),lvn_pos))

                      ! rotate distance vectors into local lat-lon coordinates:
                      !
                      ! component in longitudinal direction
                      distv_bary_1 =                                                                     &
                            z_ntdistv_bary_1*MERGE(p_patch%edges%primal_normal_cell(je,jb,1)%v1,         &
                                                   p_patch%edges%primal_normal_cell(je,jb,2)%v1,lvn_pos) &
                          + z_ntdistv_bary_2*MERGE(p_patch%edges%dual_normal_cell(je,jb,1)%v1,           &
                                                   p_patch%edges%dual_normal_cell(je,jb,2)%v1,lvn_pos)

                      ! component in latitudinal direction
                      distv_bary_2 =                                                                     &
                            z_ntdistv_bary_1*MERGE(p_patch%edges%primal_normal_cell(je,jb,1)%v2,         &
                                                   p_patch%edges%primal_normal_cell(je,jb,2)%v2,lvn_pos) &
                          + z_ntdistv_bary_2*MERGE(p_patch%edges%dual_normal_cell(je,jb,1)%v2,           &
                                                   p_patch%edges%dual_normal_cell(je,jb,2)%v2,lvn_pos)


                      ! Calculate "edge values" of rho and theta_v
                      ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                      ! and the corresponding gradients are stored in z_grad_rth.
    #ifdef __SWAPDIM
                      z_rho_e(je,jk,jb) =                                                     &
                        REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,1) &
                        + distv_bary_1 * z_grad_rth(ilc0,jk,ibc0,1) &
                        + distv_bary_2 * z_grad_rth(ilc0,jk,ibc0,2)
                      z_theta_v_e(je,jk,jb) =                                                   &
                        REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,2) &
                        + distv_bary_1 * z_grad_rth(ilc0,jk,ibc0,3)                             &
                        + distv_bary_2 * z_grad_rth(ilc0,jk,ibc0,4)
    #else
                      z_rho_e(je,jk,jb) = REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) &
                        +                      z_rth_pr(1,ilc0,jk,ibc0)              &
                        + distv_bary_1 * z_grad_rth(1,ilc0,jk,ibc0)                  &
                        + distv_bary_2 * z_grad_rth(2,ilc0,jk,ibc0)

                      z_theta_v_e(je,jk,jb) = REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) &
                        +                          z_rth_pr(2,ilc0,jk,ibc0)                &
                        + distv_bary_1 * z_grad_rth(3,ilc0,jk,ibc0)                        &
                        + distv_bary_2 * z_grad_rth(4,ilc0,jk,ibc0)
    #endif
                    ENDDO   ! loop over vertical levels
                  ENDDO ! loop over edges
    #else
                  !$ACC LOOP GANG VECTOR COLLAPSE(2) PRIVATE(ilc0,ibc0)
                  DO jk = 1, nlev
                    DO je = i_startidx, i_endidx

                      ilc0 = p_cell_idx(je,jk,jb)
                      ibc0 = p_cell_blk(je,jk,jb)

                      ! Calculate "edge values" of rho and theta_v
                      ! Note: z_rth_pr contains the perturbation values of rho and theta_v,
                      ! and the corresponding gradients are stored in z_grad_rth.
    #ifdef __SWAPDIM
                      z_rho_e(je,jk,jb) =                                                       &
                        REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,1)   &
                        + p_distv_bary(je,jk,jb,1) * z_grad_rth(ilc0,jk,ibc0,1)             &
                        + p_distv_bary(je,jk,jb,2) * z_grad_rth(ilc0,jk,ibc0,2)
                      z_theta_v_e(je,jk,jb) =                                                   &
                        REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) + z_rth_pr(ilc0,jk,ibc0,2) &
                        + p_distv_bary(je,jk,jb,1) * z_grad_rth(ilc0,jk,ibc0,3)             &
                        + p_distv_bary(je,jk,jb,2) * z_grad_rth(ilc0,jk,ibc0,4)
    #else
                      z_rho_e(je,jk,jb) = REAL(p_nh%metrics%rho_ref_me(je,jk,jb),wp)     &
                        +                            z_rth_pr(1,ilc0,jk,ibc0)            &
                        + p_distv_bary(je,jk,jb,1) * z_grad_rth(1,ilc0,jk,ibc0)      &
                        + p_distv_bary(je,jk,jb,2) * z_grad_rth(2,ilc0,jk,ibc0)
                      z_theta_v_e(je,jk,jb) = REAL(p_nh%metrics%theta_ref_me(je,jk,jb),wp) &
                        +                            z_rth_pr(2,ilc0,jk,ibc0)              &
                        + p_distv_bary(je,jk,jb,1) * z_grad_rth(3,ilc0,jk,ibc0)        &
                        + p_distv_bary(je,jk,jb,2) * z_grad_rth(4,ilc0,jk,ibc0)
    #endif

                    ENDDO ! loop over edges
                  ENDDO   ! loop over vertical levels
    #endif
    !$ACC END PARALLEL
    !$DSL END STENCIL(name=mo_solve_nonhydro_stencil_16)


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
    !$DSL END STENCIL(name=calculate_nabla2_for_w; noendif=True)
    !$DSL END CREATE()
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
