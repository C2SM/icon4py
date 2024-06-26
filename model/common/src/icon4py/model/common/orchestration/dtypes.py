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
import dace

# Define DaCe Symbols
CellDim_sym = dace.symbol('CellDim_sym')
EdgeDim_sym = dace.symbol('EdgeDim_sym')
VertexDim_sym = dace.symbol('VertexDim_sym')
KDim_sym = dace.symbol('KDim_sym')

# Symbols for the strides
DiffusionDiagnosticState_hdef_ic_s0_sym = dace.symbol('DiffusionDiagnosticState_hdef_ic_s0_sym')
DiffusionDiagnosticState_hdef_ic_s1_sym = dace.symbol('DiffusionDiagnosticState_hdef_ic_s1_sym')
DiffusionDiagnosticState_div_ic_s0_sym = dace.symbol('DiffusionDiagnosticState_div_ic_s0_sym')
DiffusionDiagnosticState_div_ic_s1_sym = dace.symbol('DiffusionDiagnosticState_div_ic_s1_sym')
DiffusionDiagnosticState_dwdx_s0_sym = dace.symbol('DiffusionDiagnosticState_dwdx_s0_sym')
DiffusionDiagnosticState_dwdx_s1_sym = dace.symbol('DiffusionDiagnosticState_dwdx_s1_sym')
DiffusionDiagnosticState_dwdy_s0_sym = dace.symbol('DiffusionDiagnosticState_dwdy_s0_sym')
DiffusionDiagnosticState_dwdy_s1_sym = dace.symbol('DiffusionDiagnosticState_dwdy_s1_sym')

PrognosticState_rho_s0_sym = dace.symbol('PrognosticState_rho_s0_sym')
PrognosticState_rho_s1_sym = dace.symbol('PrognosticState_rho_s1_sym')
PrognosticState_w_s0_sym = dace.symbol('PrognosticState_w_s0_sym')
PrognosticState_w_s1_sym = dace.symbol('PrognosticState_w_s1_sym')
PrognosticState_vn_s0_sym = dace.symbol('PrognosticState_vn_s0_sym')
PrognosticState_vn_s1_sym = dace.symbol('PrognosticState_vn_s1_sym')
PrognosticState_exner_s0_sym = dace.symbol('PrognosticState_exner_s0_sym')
PrognosticState_exner_s1_sym = dace.symbol('PrognosticState_exner_s1_sym')
PrognosticState_theta_v_s0_sym = dace.symbol('PrognosticState_theta_v_s0_sym')
PrognosticState_theta_v_s1_sym = dace.symbol('PrognosticState_theta_v_s1_sym')

DiffusionDiagnosticState_dace_t = dace.data.Structure(dict(hdef_ic=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_hdef_ic_s0_sym, DiffusionDiagnosticState_hdef_ic_s1_sym]),
                                                           div_ic=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_div_ic_s0_sym, DiffusionDiagnosticState_div_ic_s1_sym]),
                                                           dwdx=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_dwdx_s0_sym, DiffusionDiagnosticState_dwdx_s1_sym]),
                                                           dwdy=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_dwdy_s0_sym, DiffusionDiagnosticState_dwdy_s1_sym])),
                                                      name = 'DiffusionDiagnosticState')

PrognosticState_dace_t = dace.data.Structure(dict(rho=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_rho_s0_sym, PrognosticState_rho_s1_sym]),
                                                  w=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_w_s0_sym, PrognosticState_w_s1_sym]),
                                                  vn=dace.data.Array(dtype=dace.float64, shape=[EdgeDim_sym, KDim_sym], strides=[PrognosticState_vn_s0_sym, PrognosticState_vn_s1_sym]),
                                                  exner=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_exner_s0_sym, PrognosticState_exner_s1_sym]),
                                                  theta_v=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_theta_v_s0_sym, PrognosticState_theta_v_s1_sym])),
                                             name = 'PrognosticState')
