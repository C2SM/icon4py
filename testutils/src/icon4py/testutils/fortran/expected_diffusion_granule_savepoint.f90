
    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_primal_normal_vert_v1

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_primal_normal_vert_v2

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_dual_normal_vert_v1

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_dual_normal_vert_v2

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_primal_normal_cell_v1

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_primal_normal_cell_v2

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_dual_normal_cell_v1

    !$ser verbatim real, dimension(:,:,:), allocatable :: edges_dual_normal_cell_v2

    !$ser init directory="." prefix="test"

    !$ser savepoint diffusion_init_in

    !$ser verbatim allocate(edges_primal_normal_vert_v1(size(edges_primal_normal_vert, 1), size(edges_primal_normal_vert, 2), size(edges_primal_normal_vert, 3)))
    !$ser data edges_primal_normal_vert_v1=edges_primal_normal_vert_v1(:,:,:)
    !$ser verbatim deallocate(edges_primal_normal_vert_v1)

    !$ser verbatim allocate(edges_primal_normal_vert_v2(size(edges_primal_normal_vert, 1), size(edges_primal_normal_vert, 2), size(edges_primal_normal_vert, 3)))
    !$ser data edges_primal_normal_vert_v2=edges_primal_normal_vert_v2(:,:,:)
    !$ser verbatim deallocate(edges_primal_normal_vert_v2)

    !$ser verbatim allocate(edges_dual_normal_vert_v1(size(edges_dual_normal_vert, 1), size(edges_dual_normal_vert, 2), size(edges_dual_normal_vert, 3)))
    !$ser data edges_dual_normal_vert_v1=edges_dual_normal_vert_v1(:,:,:)
    !$ser verbatim deallocate(edges_dual_normal_vert_v1)

    !$ser verbatim allocate(edges_dual_normal_vert_v2(size(edges_dual_normal_vert, 1), size(edges_dual_normal_vert, 2), size(edges_dual_normal_vert, 3)))
    !$ser data edges_dual_normal_vert_v2=edges_dual_normal_vert_v2(:,:,:)
    !$ser verbatim deallocate(edges_dual_normal_vert_v2)

    !$ser verbatim allocate(edges_primal_normal_cell_v1(size(edges_primal_normal_cell, 1), size(edges_primal_normal_cell, 2), size(edges_primal_normal_cell, 3)))
    !$ser data edges_primal_normal_cell_v1=edges_primal_normal_cell_v1(:,:,:)
    !$ser verbatim deallocate(edges_primal_normal_cell_v1)

    !$ser verbatim allocate(edges_primal_normal_cell_v2(size(edges_primal_normal_cell, 1), size(edges_primal_normal_cell, 2), size(edges_primal_normal_cell, 3)))
    !$ser data edges_primal_normal_cell_v2=edges_primal_normal_cell_v2(:,:,:)
    !$ser verbatim deallocate(edges_primal_normal_cell_v2)

    !$ser verbatim allocate(edges_dual_normal_cell_v1(size(edges_dual_normal_cell, 1), size(edges_dual_normal_cell, 2), size(edges_dual_normal_cell, 3)))
    !$ser data edges_dual_normal_cell_v1=edges_dual_normal_cell_v1(:,:,:)
    !$ser verbatim deallocate(edges_dual_normal_cell_v1)

    !$ser verbatim allocate(edges_dual_normal_cell_v2(size(edges_dual_normal_cell, 1), size(edges_dual_normal_cell, 2), size(edges_dual_normal_cell, 3)))
    !$ser data edges_dual_normal_cell_v2=edges_dual_normal_cell_v2(:,:,:)
    !$ser verbatim deallocate(edges_dual_normal_cell_v2)

    !$ser data cvd_o_rd=cvd_o_rd

    !$ser data grav=grav

    !$ser data jg=jg

    !$ser data nproma=nproma

    !$ser data nlev=nlev

    !$ser data nblks_e=nblks_e

    !$ser data nblks_v=nblks_v

    !$ser data nblks_c=nblks_c

    !$ser data nshift=nshift

    !$ser data nshift_total=nshift_total

    !$ser data nrdmax=nrdmax

    !$ser data ndyn_substeps=ndyn_substeps

    !$ser data hdiff_order=hdiff_order

    !$ser data itype_comm=itype_comm

    !$ser data itype_sher=itype_sher

    !$ser data itype_vn_diffu=itype_vn_diffu

    !$ser data itype_t_diffu=itype_t_diffu

    !$ser data hdiff_smag_z=hdiff_smag_z

    !$ser data hdiff_smag_z2=hdiff_smag_z2

    !$ser data hdiff_smag_z3=hdiff_smag_z3

    !$ser data hdiff_smag_z4=hdiff_smag_z4

    !$ser data hdiff_smag_fac=hdiff_smag_fac

    !$ser data hdiff_smag_fac2=hdiff_smag_fac2

    !$ser data hdiff_smag_fac3=hdiff_smag_fac3

    !$ser data hdiff_smag_fac4=hdiff_smag_fac4

    !$ser data hdiff_efdt_ratio=hdiff_efdt_ratio

    !$ser data k4=k4

    !$ser data k4w=k4w

    !$ser data nudge_max_coeff=nudge_max_coeff

    !$ser data denom_diffu_v=denom_diffu_v

    !$ser data p_test_run=p_test_run

    !$ser data lphys=lphys

    !$ser data lhdiff_rcf=lhdiff_rcf

    !$ser data lhdiff_w=lhdiff_w

    !$ser data lhdiff_temp=lhdiff_temp

    !$ser data l_zdiffu_t=l_zdiffu_t

    !$ser data l_limited_area=l_limited_area

    !$ser data lfeedback=lfeedback

    !$ser data ltkeshs=ltkeshs

    !$ser data lsmag_3d=lsmag_3d

    !$ser data lvert_nest=lvert_nest

    !$ser data ltimer=ltimer

    !$ser data ddt_vn_hdf_is_associated=ddt_vn_hdf_is_associated

    !$ser data ddt_vn_dyn_is_associated=ddt_vn_dyn_is_associated

    IF (SIZE(vct_a) > 0) THEN
       !$ser data vct_a=vct_a(:)
    ELSE
       PRINT *, 'Warning: Array vct_a has size 0. Not serializing array.'
    END IF

    IF (SIZE(c_lin_e) > 0) THEN
       !$ser data c_lin_e=c_lin_e(:,:,:)
    ELSE
       PRINT *, 'Warning: Array c_lin_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(e_bln_c_s) > 0) THEN
       !$ser data e_bln_c_s=e_bln_c_s(:,:,:)
    ELSE
       PRINT *, 'Warning: Array e_bln_c_s has size 0. Not serializing array.'
    END IF

    IF (SIZE(e_bln_c_u) > 0) THEN
       !$ser data e_bln_c_u=e_bln_c_u(:,:,:)
    ELSE
       PRINT *, 'Warning: Array e_bln_c_u has size 0. Not serializing array.'
    END IF

    IF (SIZE(e_bln_c_v) > 0) THEN
       !$ser data e_bln_c_v=e_bln_c_v(:,:,:)
    ELSE
       PRINT *, 'Warning: Array e_bln_c_v has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_aw_verts) > 0) THEN
       !$ser data cells_aw_verts=cells_aw_verts(:,:,:)
    ELSE
       PRINT *, 'Warning: Array cells_aw_verts has size 0. Not serializing array.'
    END IF

    IF (SIZE(geofac_div) > 0) THEN
       !$ser data geofac_div=geofac_div(:,:,:)
    ELSE
       PRINT *, 'Warning: Array geofac_div has size 0. Not serializing array.'
    END IF

    IF (SIZE(geofac_rot) > 0) THEN
       !$ser data geofac_rot=geofac_rot(:,:,:)
    ELSE
       PRINT *, 'Warning: Array geofac_rot has size 0. Not serializing array.'
    END IF

    IF (SIZE(geofac_n2s) > 0) THEN
       !$ser data geofac_n2s=geofac_n2s(:,:,:)
    ELSE
       PRINT *, 'Warning: Array geofac_n2s has size 0. Not serializing array.'
    END IF

    IF (SIZE(geofac_grg) > 0) THEN
       !$ser data geofac_grg=geofac_grg(:,:,:,:)
    ELSE
       PRINT *, 'Warning: Array geofac_grg has size 0. Not serializing array.'
    END IF

    IF (SIZE(nudgecoeff_e) > 0) THEN
       !$ser data nudgecoeff_e=nudgecoeff_e(:,:)
    ELSE
       PRINT *, 'Warning: Array nudgecoeff_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(rbf_vec_idx_v) > 0) THEN
       !$ser data rbf_vec_idx_v=rbf_vec_idx_v(:,:,:)
    ELSE
       PRINT *, 'Warning: Array rbf_vec_idx_v has size 0. Not serializing array.'
    END IF

    IF (SIZE(rbf_vec_blk_v) > 0) THEN
       !$ser data rbf_vec_blk_v=rbf_vec_blk_v(:,:,:)
    ELSE
       PRINT *, 'Warning: Array rbf_vec_blk_v has size 0. Not serializing array.'
    END IF

    IF (SIZE(rbf_vec_coeff_v) > 0) THEN
       !$ser data rbf_vec_coeff_v=rbf_vec_coeff_v(:,:,:,:)
    ELSE
       PRINT *, 'Warning: Array rbf_vec_coeff_v has size 0. Not serializing array.'
    END IF

    IF (SIZE(enhfac_diffu) > 0) THEN
       !$ser data enhfac_diffu=enhfac_diffu(:)
    ELSE
       PRINT *, 'Warning: Array enhfac_diffu has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_intcoef) > 0) THEN
       !$ser data zd_intcoef=zd_intcoef(:,:)
    ELSE
       PRINT *, 'Warning: Array zd_intcoef has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_geofac) > 0) THEN
       !$ser data zd_geofac=zd_geofac(:,:)
    ELSE
       PRINT *, 'Warning: Array zd_geofac has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_diffcoef) > 0) THEN
       !$ser data zd_diffcoef=zd_diffcoef(:)
    ELSE
       PRINT *, 'Warning: Array zd_diffcoef has size 0. Not serializing array.'
    END IF

    IF (SIZE(wgtfac_c) > 0) THEN
       !$ser data wgtfac_c=wgtfac_c(:,:,:)
    ELSE
       PRINT *, 'Warning: Array wgtfac_c has size 0. Not serializing array.'
    END IF

    IF (SIZE(wgtfac_e) > 0) THEN
       !$ser data wgtfac_e=wgtfac_e(:,:,:)
    ELSE
       PRINT *, 'Warning: Array wgtfac_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(wgtfacq_e) > 0) THEN
       !$ser data wgtfacq_e=wgtfacq_e(:,:,:)
    ELSE
       PRINT *, 'Warning: Array wgtfacq_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(wgtfacq1_e) > 0) THEN
       !$ser data wgtfacq1_e=wgtfacq1_e(:,:,:)
    ELSE
       PRINT *, 'Warning: Array wgtfacq1_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(ddqz_z_full_e) > 0) THEN
       !$ser data ddqz_z_full_e=ddqz_z_full_e(:,:,:)
    ELSE
       PRINT *, 'Warning: Array ddqz_z_full_e has size 0. Not serializing array.'
    END IF

    IF (SIZE(theta_ref_mc) > 0) THEN
       !$ser data theta_ref_mc=theta_ref_mc(:,:,:)
    ELSE
       PRINT *, 'Warning: Array theta_ref_mc has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_indlist) > 0) THEN
       !$ser data zd_indlist=zd_indlist(:,:)
    ELSE
       PRINT *, 'Warning: Array zd_indlist has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_blklist) > 0) THEN
       !$ser data zd_blklist=zd_blklist(:,:)
    ELSE
       PRINT *, 'Warning: Array zd_blklist has size 0. Not serializing array.'
    END IF

    IF (SIZE(zd_vertidx) > 0) THEN
       !$ser data zd_vertidx=zd_vertidx(:,:)
    ELSE
       PRINT *, 'Warning: Array zd_vertidx has size 0. Not serializing array.'
    END IF

    !$ser data zd_listdim=zd_listdim

    IF (SIZE(edges_start_block) > 0) THEN
       !$ser data edges_start_block=edges_start_block(min_rledge:)
    ELSE
       PRINT *, 'Warning: Array edges_start_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_end_block) > 0) THEN
       !$ser data edges_end_block=edges_end_block(min_rledge:)
    ELSE
       PRINT *, 'Warning: Array edges_end_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_start_index) > 0) THEN
       !$ser data edges_start_index=edges_start_index(min_rledge:)
    ELSE
       PRINT *, 'Warning: Array edges_start_index has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_end_index) > 0) THEN
       !$ser data edges_end_index=edges_end_index(min_rledge:)
    ELSE
       PRINT *, 'Warning: Array edges_end_index has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_vertex_idx) > 0) THEN
       !$ser data edges_vertex_idx=edges_vertex_idx(:,:,:)
    ELSE
       PRINT *, 'Warning: Array edges_vertex_idx has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_vertex_blk) > 0) THEN
       !$ser data edges_vertex_blk=edges_vertex_blk(:,:,:)
    ELSE
       PRINT *, 'Warning: Array edges_vertex_blk has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_cell_idx) > 0) THEN
       !$ser data edges_cell_idx=edges_cell_idx(:,:,:)
    ELSE
       PRINT *, 'Warning: Array edges_cell_idx has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_cell_blk) > 0) THEN
       !$ser data edges_cell_blk=edges_cell_blk(:,:,:)
    ELSE
       PRINT *, 'Warning: Array edges_cell_blk has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_tangent_orientation) > 0) THEN
       !$ser data edges_tangent_orientation=edges_tangent_orientation(:,:)
    ELSE
       PRINT *, 'Warning: Array edges_tangent_orientation has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_inv_vert_vert_length) > 0) THEN
       !$ser data edges_inv_vert_vert_length=edges_inv_vert_vert_length(:,:)
    ELSE
       PRINT *, 'Warning: Array edges_inv_vert_vert_length has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_inv_primal_edge_length) > 0) THEN
       !$ser data edges_inv_primal_edge_length=edges_inv_primal_edge_length(:,:)
    ELSE
       PRINT *, 'Warning: Array edges_inv_primal_edge_length has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_inv_dual_edge_length) > 0) THEN
       !$ser data edges_inv_dual_edge_length=edges_inv_dual_edge_length(:,:)
    ELSE
       PRINT *, 'Warning: Array edges_inv_dual_edge_length has size 0. Not serializing array.'
    END IF

    IF (SIZE(edges_area_edge) > 0) THEN
       !$ser data edges_area_edge=edges_area_edge(:,:)
    ELSE
       PRINT *, 'Warning: Array edges_area_edge has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_start_block) > 0) THEN
       !$ser data cells_start_block=cells_start_block(min_rlcell:)
    ELSE
       PRINT *, 'Warning: Array cells_start_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_end_block) > 0) THEN
       !$ser data cells_end_block=cells_end_block(min_rlcell:)
    ELSE
       PRINT *, 'Warning: Array cells_end_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_start_index) > 0) THEN
       !$ser data cells_start_index=cells_start_index(min_rlcell:)
    ELSE
       PRINT *, 'Warning: Array cells_start_index has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_end_index) > 0) THEN
       !$ser data cells_end_index=cells_end_index(min_rlcell:)
    ELSE
       PRINT *, 'Warning: Array cells_end_index has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_neighbor_idx) > 0) THEN
       !$ser data cells_neighbor_idx=cells_neighbor_idx(:,:,:)
    ELSE
       PRINT *, 'Warning: Array cells_neighbor_idx has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_neighbor_blk) > 0) THEN
       !$ser data cells_neighbor_blk=cells_neighbor_blk(:,:,:)
    ELSE
       PRINT *, 'Warning: Array cells_neighbor_blk has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_edge_idx) > 0) THEN
       !$ser data cells_edge_idx=cells_edge_idx(:,:,:)
    ELSE
       PRINT *, 'Warning: Array cells_edge_idx has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_edge_blk) > 0) THEN
       !$ser data cells_edge_blk=cells_edge_blk(:,:,:)
    ELSE
       PRINT *, 'Warning: Array cells_edge_blk has size 0. Not serializing array.'
    END IF

    IF (SIZE(cells_area) > 0) THEN
       !$ser data cells_area=cells_area(:,:)
    ELSE
       PRINT *, 'Warning: Array cells_area has size 0. Not serializing array.'
    END IF

    IF (SIZE(verts_start_block) > 0) THEN
       !$ser data verts_start_block=verts_start_block(min_rlvert:)
    ELSE
       PRINT *, 'Warning: Array verts_start_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(verts_end_block) > 0) THEN
       !$ser data verts_end_block=verts_end_block(min_rlvert:)
    ELSE
       PRINT *, 'Warning: Array verts_end_block has size 0. Not serializing array.'
    END IF

    IF (SIZE(verts_start_index) > 0) THEN
       !$ser data verts_start_index=verts_start_index(min_rlvert:)
    ELSE
       PRINT *, 'Warning: Array verts_start_index has size 0. Not serializing array.'
    END IF

    IF (SIZE(verts_end_index) > 0) THEN
       !$ser data verts_end_index=verts_end_index(min_rlvert:)
    ELSE
       PRINT *, 'Warning: Array verts_end_index has size 0. Not serializing array.'
    END IF
