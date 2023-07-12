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
"""Test graupel in standalone mode using data serialized from ICON.

GT4Py hotfix:

In _external_src/gt4py-functional/src/functional/iterator/transforms/pass_manager.py
1. Exchange L49 with: inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
2. Add "return inlined" below
"""

import os

import numpy as np
from gt4py.next.ffront.fbuiltins import (
    Field,
    float64,
    int32
)

#from icon4py.atm_phy_schemes.gscp_graupel_Ong import graupel
#from icon4py.atm_phy_schemes.gscp_graupel_Ong_noTernaryInSection7 import graupel
from icon4py.atm_phy_schemes.gscp_graupel_Ong_section10 import graupel
from icon4py.common.dimension import CellDim, KDim
#from icon4py.testutils.utils import to_icon4py_field, zero_field
from gt4py.next.iterator.embedded import index_field, np_as_located_field
#from icon4py.testutils.utils_serialbox import bcolors, field_test



def test_graupel_serialized_data():
    cell_size = 2
    k_size = 2

    dt = 2.0
    dz =np_as_located_field(CellDim,KDim)(np.ones((cell_size,k_size),dtype=float64)) 
    temperature = np_as_located_field(CellDim,KDim)(np.ones((cell_size,k_size),dtype=float64))
    pres = np_as_located_field(CellDim,KDim)(np.ones((cell_size,k_size),dtype=float64))
    rho = np_as_located_field(CellDim,KDim)(np.ones((cell_size,k_size),dtype=float64))
    qv = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qc = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qi = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qr = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qs = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qg = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qnc = np_as_located_field(CellDim,KDim)(np.ones((cell_size,k_size),dtype=float64))
    ddt_tend_t = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qv = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qc = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qi = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qr = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qs = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    ddt_tend_qg = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    pri_gsp = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    prr_gsp = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    prs_gsp = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    prg_gsp = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    qrsflux = np_as_located_field(CellDim,KDim)(np.zeros((cell_size,k_size),dtype=float64))
    l_cv = True
    lpres_pri = True
    ithermo_water = int32(1)
    ldass_lhn = True
    ldiag_ttend = True
    ldiag_qtend = True
    num_cells = int32(2)
    kstart_moist = int32(0)
    kend = int32(2)
    kend_moist = int32(2)

    np_qv = np.zeros((cell_size,k_size),dtype=float64)
    np_qc = np.zeros((cell_size,k_size),dtype=float64)
    np_qi = np.zeros((cell_size,k_size),dtype=float64)
    np_qr = np.zeros((cell_size,k_size),dtype=float64)
    np_qs = np.zeros((cell_size,k_size),dtype=float64)
    np_qg = np.zeros((cell_size,k_size),dtype=float64)
    
    # Run scheme
    '''
        
    '''
    graupel(
        dt,
        dz,
        temperature,
        pres,
        rho,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
        ddt_tend_t,
        ddt_tend_qv,
        ddt_tend_qc,
        ddt_tend_qi,
        ddt_tend_qr,
        ddt_tend_qs,
        ddt_tend_qg,
        prr_gsp,
        prs_gsp,
        pri_gsp,
        prg_gsp,
        qrsflux,
        l_cv,
        lpres_pri,
        ithermo_water,
        ldass_lhn,
        ldiag_ttend,
        ldiag_qtend,
        num_cells,
        kstart_moist,
        kend,
        kend_moist,
        offset_provider={},
    )

    assert np.allclose(qv.array(),np_qv)
    assert np.allclose(qc.array(),np_qc)
    assert np.allclose(qi.array(),np_qi)
    assert np.allclose(qr.array(),np_qr)
    assert np.allclose(qs.array(),np_qs)
    assert np.allclose(qg.array(),np_qg)
    

        