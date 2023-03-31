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

import numpy as np
import serialbox as ser
from gt4py.next.common import Dimension
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2C2VDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    KHalfDim,
    V2CDim,
    V2EDim,
    VertexDim,
)


class IconSavepoint:
    def __init__(self, sp: ser.Savepoint, ser: ser.Serializer):
        self.savepoint = sp
        self.serializer = ser

    def print_meta_info(self):
        print(self.savepoint.metainfo)

    def _get_field(self, name, *dimensions, dtype=float):
        buffer = np.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        print(f"{name} {buffer.shape}")
        if len(dimensions) < len(buffer.shape):
            return np_as_located_field(*dimensions)(buffer[0])
        return np_as_located_field(*dimensions)(buffer)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}

    def _read_int32_shift1(self, name: str):
        """
        Read a index field and shift it by -1.

        use for start indeces: the shift accounts for the zero based python
        values are converted to int32
        """
        return (self.serializer.read(name, self.savepoint) - 1).astype(int32)

    def _read_int32(self, name: str):
        """
        Read a int field by name.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to int32
        """
        return self.serializer.read(name, self.savepoint).astype(int32)

    def read_int(self, name: str):
        buffer = self.serializer.read(name, self.savepoint).astype(int)
        print(f"{name} {buffer.shape}")
        return buffer


class IconGridSavePoint(IconSavepoint):
    def vct_a(self):
        return self._get_field("vct_a", KDim)

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", EdgeDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", VertexDim, E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", VertexDim, E2C2VDim)

    def owner_mask(self):
        return self._get_field("owner_mask", CellDim, dtype=bool)

    def f_e(self):
        return self._get_field("f_e", EdgeDim)

    def cell_areas(self):
        return self._get_field("cell_areas", CellDim)

    def edge_areas(self):
        return self._get_field("edge_areas", EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", EdgeDim)

    def cells_start_index(self):
        return self._read_int32_shift1("c_start_index")

    def cells_end_index(self):
        return self._read_int32("c_end_index")

    def vertex_start_index(self):
        return self._read_int32_shift1("v_start_index")

    def vertex_end_index(self):
        return self._read_int32("v_end_index")

    def edge_start_index(self):
        return self._read_int32_shift1("e_start_index")

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self.serializer.read("e_end_index", self.savepoint)

    def print_connectivity_info(name: str, ar: np.ndarray):
        print(f" connectivity {name} {ar.shape}")

    def refin_ctrl(self, dim: Dimension):
        if dim == CellDim:
            return self.serializer.read("c_refin_ctl", self.savepoint)
        elif dim == EdgeDim:
            return self.serializer.read("e_refin_ctl", self.savepoint)
        elif dim == VertexDim:
            return self.serializer.read("v_refin_ctl", self.savepoint)
        else:
            return None

    def _get_connectivity_array(self, name: str):
        connectivity = self.serializer.read(name, self.savepoint) - 1
        print(f" connectivity {name} : {connectivity.shape}")
        return connectivity

    def c2e(self):
        return self._get_connectivity_array("c2e")

    def c2e2c(self):
        return self._get_connectivity_array("c2e2c")

    def e2c2e(self):
        return self._get_connectivity_array("e2c2e")

    def e2c(self):
        return self._get_connectivity_array("e2c")

    def e2v(self):
        # array "e2v" is actually e2c2v
        v_ = self._get_connectivity_array("e2v")[:, 0:2]
        print(f"real e2v {v_.shape}")
        return v_

    def e2c2v(self):
        # array "e2v" is actually e2c2v
        return self._get_connectivity_array("e2v")

    def v2e(self):
        return self._get_connectivity_array("v2e")

    def v2c(self):
        return self._get_connectivity_array("v2c")


class IconVelocityInitSavepoint(IconSavepoint):
    def c_lin_e(self):
        return self._get_field("c_lin_e", EdgeDim, E2CDim)

    def cfl_w_limit(self) -> float:
        return self.serializer.read("cfl_w_limit", self.savepoint)[0]

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", CellDim, KDim)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", CellDim, KDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", EdgeDim, KDim)

    def coeff_gradekin(self):
        return self._get_field("coeff_gradekin", ECDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", CellDim, KDim)

    def ddt_w_adv_pc_before(self):
        return self._get_field("ddt_w_adv_pc", CellDim, KDim)

    def ddt_vn_apc_pc_before(self):
        return self._get_field("ddt_vn_apc_pc", EdgeDim, KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", EdgeDim, KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", EdgeDim, KDim)

    def geofac_grdiv(self):
        return self._get_field("geofac_grdiv", EdgeDim, E2C2EODim)

    def rbf_vec_coeff_e(self):
        buffer = np.squeeze(
            self.serializer.read("rbf_vec_coeff_e", self.savepoint).astype(float)
        ).transpose()
        return np_as_located_field(EdgeDim, E2C2EDim)(buffer)

    def c_intp(self):
        return self._get_field("c_intp", VertexDim, V2CDim)

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", VertexDim, V2EDim)

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def vn_ie(self):
        return self._get_field("vn_ie", EdgeDim, KDim)

    def vt(self):
        return self._get_field("vt", EdgeDim, KDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def wgtfacq_e(self):
        return self._get_field("wgtfacq_e", EdgeDim, KDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", EdgeDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", EdgeDim, KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", EdgeDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)


class IconDiffusionInitSavepoint(IconSavepoint):
    def hdef_ic(self):
        return self._get_field("hdef_ic", CellDim, KDim)

    def div_ic(self):
        return self._get_field("div_ic", CellDim, KDim)

    def dwdx(self):
        return self._get_field("dwdx", CellDim, KDim)

    def dwdy(self):
        return self._get_field("dwdy", CellDim, KDim)

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)

    def exner(self):
        return self._get_field("exner", CellDim, KDim)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def mask_diff(self):
        return self._get_field("mask_hdiff", CellDim, KDim, dtype=int)

    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_intcoef(self):
        return self._get_field("vcoef", CellDim, C2E2CDim, KDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def geofac_div(self):
        return self._get_field("geofac_div", CellDim, C2EDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def geofac_grg(self):
        grg = np.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        return np_as_located_field(CellDim, C2E2CODim)(
            grg[:, :, 0]
        ), np_as_located_field(CellDim, C2E2CODim)(grg[:, :, 1])

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", EdgeDim)

    def zd_vertidx(self):
        # TODO fix this
        return self._get_field("zd_vertidx", CellDim, C2E2CDim, dtype=int)

    def zd_vertoffset(self):
        return self._get_field("zd_vertoffset", CellDim, C2E2CDim, KDim, dtype=int)

    def rbf_vec_coeff_v1(self):
        return self._get_field("rbf_vec_coeff_v1", VertexDim, V2EDim)

    def rbf_vec_coeff_v2(self):
        return self._get_field("rbf_vec_coeff_v2", VertexDim, V2EDim)

    def diff_multfac_smag(self):
        return np.squeeze(self.serializer.read("diff_multfac_smag", self.savepoint))

    def smag_limit(self):
        return np.squeeze(self.serializer.read("smag_limit", self.savepoint))

    def diff_multfac_n2w(self):
        return np.squeeze(self.serializer.read("diff_multfac_n2w", self.savepoint))

    def nudgezone_diff(self) -> int:
        return self.serializer.read("nudgezone_diff", self.savepoint)[0]

    def bdy_diff(self) -> int:
        return self.serializer.read("bdy_diff", self.savepoint)[0]

    def fac_bdydiff_v(self) -> int:
        return self.serializer.read("fac_bdydiff_v", self.savepoint)[0]

    def smag_offset(self):
        return self.serializer.read("smag_offset", self.savepoint)[0]

    def diff_multfac_w(self):
        return self.serializer.read("diff_multfac_w", self.savepoint)[0]

    def diff_multfac_vn(self):
        return self.serializer.read("diff_multfac_vn", self.savepoint)


class IconExitSavepoint(IconSavepoint):
    def vn(self):
        return self._get_field("x_vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("x_theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("x_w", CellDim, KDim)

    def exner(self):
        return self._get_field("x_exner", CellDim, KDim)

    def ddt_vn_apc_pc(self):
        return self._get_field("x_ddt_vn_apc_pc", EdgeDim, KDim)

    def ddt_w_adv_pc(self):
        return self._get_field("x_ddt_w_adv_pc", CellDim, KDim)

    def vn_ie(self):
        return self._get_field("x_vn_ie", EdgeDim, KHalfDim)

    def vt(self):
        return self._get_field("x_vt", EdgeDim, KDim)

    def z_kin_hor_e(self):
        return self._get_field("x_z_kin_hor_e", EdgeDim, KDim)

    def z_vt_ie(self):
        return self._get_field("x_z_vt_ie", EdgeDim, KDim)

    def z_w_concorr_me(self):
        return self._get_field("x_z_w_concorr_me", EdgeDim, KDim)


class IconSerialDataProvider:
    def __init__(self, fname_prefix, path=".", do_print=False):
        self.rank = 0
        self.serializer: ser.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{str(self.rank)}"
        self._init_serializer(do_print)

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            print(" WARNING: no filename! closing serializer")
        self.serializer = ser.Serializer(
            ser.OpenModeKind.Read, str(self.file_path), self.fname
        )
        if do_print:
            self.print_info()

    def print_info(self):
        print(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        print(f"FIELDNAMES: {self.serializer.fieldnames()}")

    def from_savepoint_grid(self) -> IconGridSavePoint:
        savepoint = self.serializer.savepoint["icon-grid"].id[1].as_savepoint()
        return IconGridSavePoint(savepoint, self.serializer)

    def from_savepoint_diffusion_init(
        self, linit: bool, date: str
    ) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconDiffusionInitSavepoint(savepoint, self.serializer)

    def from_savepoint_velocity_init(
        self, istep: int, vn_only: bool, date: str, jstep: int
    ) -> IconVelocityInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-velocity-tendencies"]
            .istep[istep]
            .vn_only[vn_only]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconVelocityInitSavepoint(savepoint, self.serializer)

    def from_savepoint_diffusion_exit(
        self, linit: bool, date: str
    ) -> IconExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-exit"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconExitSavepoint(savepoint, self.serializer)

    def from_savepoint_velocity_exit(
        self, istep: int, vn_only: bool, date: str, jstep: int
    ) -> IconExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-velocity-tendencies"]
            .istep[istep]
            .vn_only[vn_only]
            .date[date]
            .jstep[jstep]
            .as_savepoint()
        )
        return IconExitSavepoint(savepoint, self.serializer)
