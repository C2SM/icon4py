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


import contextlib
import io
import trace

import numpy as np
from hypothesis import given, settings, target

from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.atm_phy_schemes.mo_satad import _newtonian_iteration_temp, satad, _satad
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import phy_const
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh
from icon4py.model.common.test_utils.helpers import maximizeTendency, random_field_strategy

import os
import sys
import serialbox as ser
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import (
    run_gtfn,
    run_gtfn_cached,
    run_gtfn_imperative,
)

cp_v = 1850.0
ci = 2108.0

tol = 1e-3
maxiter = 10  # DL: Needs to be 10, but GT4Py is currently too slow
zqwmin = 1e-20


def latent_heat_vaporization(t):
    """Return latent heat of vaporization given a temperature."""
    return (
        phy_const.alv
        + (cp_v - phy_const.clw) * (t - phy_const.tmelt)
        - phy_const.rv * t
    )


def sat_pres_water(t):
    """Saturation pressure of water."""
    return conv_table.c1es * np.exp(
        conv_table.c3les * (t - phy_const.tmelt) / (t - conv_table.c4les)
    )


def qsat_rho(t, rho):
    return sat_pres_water(t) / (rho * phy_const.rv * t)


def dqsatdT_rho(t, zqsat):
    """Return derivative of qsat with respect to t."""
    beta = conv_table.c5les / (t - conv_table.c4les) ** 2 - 1.0 / t
    return beta * zqsat


def newtonian_iteration_temp(t, twork, tworkold, qv, rho):
    """Obtain temperature at saturation using Newtonian iteration."""
    lwdocvd = latent_heat_vaporization(t) / phy_const.cvd

    for ctr in range(maxiter):
        if abs(twork - tworkold) > tol:
            tworkold = twork
            qwd = qsat_rho(twork, rho)
            dqwd = dqsatdT_rho(twork, qwd)
            fT = twork - t + lwdocvd * (qwd - qv)
            dfT = 1.0 + lwdocvd * dqwd
            twork = twork - fT / dfT

    return twork


def satad_numpy(qv, qc, t, rho):
    """Numpy translation of satad_v_3D from Fortan ICON."""
    for cell, k in np.ndindex(np.shape(qv)):
        lwdocvd = latent_heat_vaporization(t[cell, k]) / phy_const.cvd

        Ttest = t[cell, k] - lwdocvd * qc[cell, k]

        if qv[cell, k] + qc[cell, k] <= qsat_rho(Ttest, rho[cell, k]):
            qv[cell, k] = qv[cell, k] + qc[cell, k]
            qc[cell, k] = 0.0
            t[cell, k] = Ttest
        else:
            t[cell, k] = newtonian_iteration_temp(
                t[cell, k], t[cell, k], t[cell, k] + 10.0, qv[cell, k], rho[cell, k]
            )

            qwa = qsat_rho(t[cell, k], rho[cell, k])
            qc[cell, k] = max(qc[cell, k] + qv[cell, k] - qwa, zqwmin)
            qv[cell, k] = qwa

    return t, qv, qc


def newtonian_iteration_temp_Ong(t, twork, tworkold, qv, rho):
    """Obtain temperature at saturation using Newtonian iteration."""
    lwdocvd = latent_heat_vaporization(t) / phy_const.cvd

    ctr_real = 0
    for ctr in range(maxiter):
        if abs(twork - tworkold) > tol:
            tworkold = twork
            qwd = qsat_rho(twork, rho)
            dqwd = dqsatdT_rho(twork, qwd)
            fT = twork - t + lwdocvd * (qwd - qv)
            dfT = 1.0 + lwdocvd * dqwd
            twork = twork - fT / dfT
            ctr_real = ctr_real + 1

    return twork, ctr_real, abs(twork - tworkold)


def satad_numpy_Ong(qv, qc, t, rho):
    """Numpy translation of satad_v_3D from Fortan ICON."""
    ctr_array = np.full(qv.shape,fill_value=-1,dtype=int)
    diff_array = np.zeros(qv.shape, dtype=float)
    for cell, k in np.ndindex(np.shape(qv)):
        lwdocvd = latent_heat_vaporization(t[cell, k]) / phy_const.cvd

        Ttest = t[cell, k] - lwdocvd * qc[cell, k]

        if qv[cell, k] + qc[cell, k] <= qsat_rho(Ttest, rho[cell, k]):
            qv[cell, k] = qv[cell, k] + qc[cell, k]
            qc[cell, k] = 0.0
            t[cell, k] = Ttest
        else:
            t[cell, k], ctr_array[cell, k], diff_array[cell, k] = newtonian_iteration_temp_Ong(
                t[cell, k], t[cell, k], t[cell, k] + 10.0, qv[cell, k], rho[cell, k]
            )

            qwa = qsat_rho(t[cell, k], rho[cell, k])
            qc[cell, k] = max(qc[cell, k] + qv[cell, k] - qwa, zqwmin)
            qv[cell, k] = qwa

    return t, qv, qc, ctr_array, diff_array

# TODO: Understand magic number 1e-8. Single precision-related?
@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=200, max_value=350),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
)
@settings(deadline=None, max_examples=10)
def nontest_newtonian_iteration(t, qv, rho):
    """Test newtonian_iteration aginst a numpy implementaion."""
    tRef = np.zeros_like(np.asarray(t))

    # Numpy Implementation
    for cell, k in np.ndindex(np.shape(t)):

        tRef[cell, k] = newtonian_iteration_temp(
            np.asarray(t)[cell, k],
            np.asarray(t)[cell, k],
            np.asarray(t)[cell, k] + 10.0,
            np.asarray(qv)[cell, k],
            np.asarray(rho)[cell, k],
        )

    # Guide hypothesis tool to maximize tendency of t
    maximizeTendency(t, tRef, "t")

    # GT4Py Implementation
    _newtonian_iteration_temp(t, qv, rho, out=t, offset_provider={})

    assert np.allclose(np.asarray(t), tRef)


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=200, max_value=350),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
)
@settings(deadline=None, max_examples=10)
def nontest_mo_satad(qv, qc, t, rho):
    """Test satad aginst a numpy implementaion."""
    # Numpy Implementation
    tracer = trace.Trace(trace=1, count=1)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        tRef, qvRef, qcRef = tracer.runfunc(
            satad_numpy,
            np.asarray(qv).copy(),
            np.asarray(qc).copy(),
            np.asarray(t).copy(),
            np.asarray(rho).copy(),
        )

    lines = len(
        [line_no for fname, line_no in tracer.counts.keys() if fname == __file__]
    )

    # Guide hypothesis to maximize the number of lines visited
    target(lines, label="lines")

    # Guide hypothesis tool to maximize tendencies
    maximizeTendency(t, tRef, "t")
    maximizeTendency(qv, qvRef, "qv")
    maximizeTendency(qc, qcRef, "qc")

    # GT4Py Implementation
    satad(qv, qc, t, rho, offset_provider={})

    # Check results using a tolerance test
    assert np.allclose(np.asarray(t), tRef)
    assert np.allclose(np.asarray(qv), qvRef)
    assert np.allclose(np.asarray(qc), qcRef)


def test_serialize_mo_satad():
    """Test satad aginst a numpy implementaion."""

    mpi_ranks = np.arange(0, 10, dtype=int)
    initial_date = "2008-09-01T00:00:00.000"
    dates = ("2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000")
    Nblocks = 2  # 121
    rank = mpi_ranks[7]
    blocks = tuple(i + 1 for i in range(Nblocks))
    print(dates)
    print(blocks)

    # please put the data in the serialbox directory
    script_dir = os.path.dirname(__file__)
    base_dir = script_dir + '/serialbox/data_dir/'
    # base_dir = "/home/ong/Data/nh_wk_rerun_complete/data_dir/"
    try:
        serializer = ser.Serializer(ser.OpenModeKind.Read, base_dir, "wk__rank" + str(rank))
        savePoints = serializer.savepoint_list()
    except ser.SerialboxError as e:
        print(f"serializer: error: {e}")
        print("Data download link: https://polybox.ethz.ch/index.php/s/LEh6pZ9etDvNO0c")
        sys.exit(1)

    ser_field_name = (
        "ser_graupel_temperature",
        "ser_graupel_pres",
        "ser_graupel_rho",
        "ser_graupel_qv",
        "ser_graupel_qc",
    )

    field_name = (
        "temperature",
        "pres",
        "rho",
        "qv",
        "qc",
    )

    # construct serialized data dictionary
    ser_data = {}
    for item in ser_field_name:
        ser_data[item] = None

    # set date
    date_index = 0

    # read serialized input and reference data
    for item_no, item in enumerate(ser_field_name):
        exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[0]]["date"][dates[date_index]]
        ser_data[field_name[item_no]] = serializer.read(item, exit_savePoint)
        for i in range(Nblocks - 1):
            exit_savePoint = serializer.savepoint["call-graupel-exit"]["serial_state"][1]["block_index"][blocks[i + 1]]["date"][dates[date_index]]
            ser_data[field_name[item_no]] = np.concatenate((ser_data[field_name[item_no]], serializer.read(item, exit_savePoint)), axis=0)

    # Numpy Implementation
    tRef, qvRef, qcRef, ctrRef, diffRef = satad_numpy_Ong(
        ser_data["qv"].copy(),
        ser_data["qc"].copy(),
        ser_data["temperature"].copy(),
        ser_data["rho"].copy(),
    )

    (cell_size, k_size) = tRef.shape
    j = 0
    for i in range(cell_size):
        for k in range(k_size):
            if (ctrRef[i,k] > 0):
                j = j + 1
                '''
                print("diff: {0:d}, {1:.10e}, {2:.10e}, {3:.10e}, {4:.10e}".format(
                    ctrRef[i,k],
                    tRef[i,k] - ser_data['temperature'][i,k],
                    qvRef[i,k] - ser_data['qv'][i,k],
                    qcRef[i,k] - ser_data['qc'][i,k],
                    diffRef[i,k])
                )
                print("new value: {0:d}, {1:.10e}, {2:.10e}, {3:.10e}".format(
                    ctrRef[i,k],
                    tRef[i,k],
                    qvRef[i,k],
                    qcRef[i,k])
                )
                print()
                '''
    print("total adjustment: ", j, float(j)/float(cell_size)/float(k_size))

    ser_field = {}
    for item in field_name:
        ser_field[item] = np_as_located_field(CellDim, KDim)(np.array(ser_data[item], dtype=float))

    satad(ser_field['qv'], ser_field['qc'], ser_field['temperature'], ser_field['rho'], offset_provider={})
    q = 0
    for i in range(cell_size):
        for k in range(k_size):
            if (abs(ser_field['temperature'].array()[i,k] - tRef[i,k]) > 1.e-10):
                q = q + 1
                print (ser_field['temperature'].array()[i,k] - tRef[i,k], ser_field['temperature'].array()[i,k], tRef[i,k])

    print("total error number: ", q)
