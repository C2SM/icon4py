#!/usr/bin/env python
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import time

import gt4py.next as gtx
import numpy as np


try:
    from netCDF4 import Dataset
except ImportError:
    print("Netcdf not installed")
    sys.exit()


from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import (
    graupel_run,
)
from icon4py.model.common import dimension as dims, model_backends


def set_lib_path(lib_dir):
    sys.path.append(lib_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        metavar="output_file",
        dest="output_file",
        help="output filename",
        default="output.nc",
    )
    parser.add_argument("input_file", help="input data file")
    parser.add_argument("itime", help="time-index", nargs="?", default=0)
    parser.add_argument("dt", help="timestep", nargs="?", default=30.0)
    parser.add_argument("qnc", help="Water number concentration", nargs="?", default=100.0)
    parser.add_argument(
        "-ldir",
        metavar="lib_dir",
        dest="ldir",
        help="directory with py_graupel shared lib",
        default="build/lib64",
    )
    parser.add_argument(
        "-with_sat_adj",
        action="store_true",
    )

    return parser.parse_args()


class Data:
    def __init__(self, args):
        nc = Dataset(args.input_file)
        # intent(in) variables:
        try:
            self.ncells = len(nc.dimensions["cell"])
        except KeyError:
            self.ncells = len(nc.dimensions["ncells"])

        self.nlev = len(nc.dimensions["height"])
        self.z = nc.variables["zg"][:, :].astype(np.float64)
        self.p = nc.variables["pfull"][:, :].astype(np.float64)
        self.rho = nc.variables["rho"][:, :].astype(np.float64)
        # intent(inout) variables:
        self.t = nc.variables["ta"][:, :].astype(np.float64)  # inout
        self.qv = nc.variables["hus"][:, :].astype(np.float64)  # inout
        self.qc = nc.variables["clw"][:, :].astype(np.float64)  # inout
        self.qi = nc.variables["cli"][:, :].astype(np.float64)  # inout
        self.qr = nc.variables["qr"][:, :].astype(np.float64)  # inout
        self.qs = nc.variables["qs"][:, :].astype(np.float64)  # inout
        self.qg = nc.variables["qg"][:, :].astype(np.float64)  # inout
        # intent(out) variables:
        self.t_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qv_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qc_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qi_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qr_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qs_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.qg_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.pflx_out = np.zeros((self.ncells, self.nlev), np.float64)
        self.prr_gsp = np.zeros(self.ncells, np.float64)
        self.pri_gsp = np.zeros(self.ncells, np.float64)
        self.prs_gsp = np.zeros(self.ncells, np.float64)
        self.prg_gsp = np.zeros(self.ncells, np.float64)
        self.pre_gsp = np.zeros(self.ncells, np.float64)
        self.dz = calc_dz(self.nlev, self.z)
        self.mask_out = np.full((self.ncells, self.nlev), True)


def calc_dz(ksize, z):
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def write_fields(
    output_filename,
    ncell,
    nlev,
    t,
    qv,
    qc,
    qi,
    qr,
    qs,
    qg,
    prr_gsp,
    prs_gsp,
    pri_gsp,
    prg_gsp,
    pflx,
    pre_gsp,
):
    ncfile  = Dataset(output_filename, mode="w")
    ncells  = ncfile.createDimension("ncells", ncell)
    height  = ncfile.createDimension("height", nlev)
    height1 = ncfile.createDimension("height1", nlev+1)
    ta_var  = ncfile.createVariable("ta", np.double, ("height", "ncells"))
    hus_var = ncfile.createVariable("hus", np.double, ("height", "ncells"))
    clw_var = ncfile.createVariable("clw", np.double, ("height", "ncells"))
    cli_var = ncfile.createVariable("cli", np.double, ("height", "ncells"))
    qr_var  = ncfile.createVariable("qr", np.double, ("height", "ncells"))
    qs_var  = ncfile.createVariable("qs", np.double, ("height", "ncells"))
    qg_var  = ncfile.createVariable("qg", np.double, ("height", "ncells"))
    pflx_var = ncfile.createVariable("pflx", np.double, ("height", "ncells"))
    prr_gsp_var = ncfile.createVariable("prr_gsp", np.double, ("height1", "ncells"))
    prs_gsp_var = ncfile.createVariable("prs_gsp", np.double, ("height1", "ncells"))
    pri_gsp_var = ncfile.createVariable("pri_gsp", np.double, ("height1", "ncells"))
    prg_gsp_var = ncfile.createVariable("prg_gsp", np.double, ("height1", "ncells"))
    pre_gsp_var = ncfile.createVariable("pre_gsp", np.double, ("height1", "ncells"))

    ta_var[:, :] = t
    hus_var[:, :] = qv
    clw_var[:, :] = qc
    cli_var[:, :] = qi
    qr_var[:, :] = qr
    qs_var[:, :] = qs
    qg_var[:, :] = qg
    pflx_var[:, :] = pflx
    prr_gsp_var[:, :] = prr_gsp
    prs_gsp_var[:, :] = prs_gsp
    pri_gsp_var[:, :] = pri_gsp
    prg_gsp_var[:, :] = prg_gsp
    pre_gsp_var[:, :] = pre_gsp
    ncfile.close()

backend = model_backends.BACKENDS["dace_gpu"]
args = get_args()

set_lib_path(args.ldir)
sys.setrecursionlimit(10**4)

data = Data(args)

t_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.t_out,
    allocator=backend,
)
qv_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qv_out,
    allocator=backend,
)
qc_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qc_out,
    allocator=backend,
)
qr_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qr_out,
    allocator=backend,
)
qs_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qs_out,
    allocator=backend,
)
qi_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qi_out,
    allocator=backend,
)
qg_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.qg_out,
    allocator=backend,
)
pflx_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.pflx_out,
    allocator=backend,
)
pr_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    np.zeros((data.ncells, data.nlev)),
    allocator=backend,
)
ps_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    np.zeros((data.ncells, data.nlev)),
    allocator=backend,
)
pi_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    np.zeros((data.ncells, data.nlev)),
    allocator=backend,
)
pg_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    np.zeros((data.ncells, data.nlev)),
    allocator=backend,
)
pre_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    np.zeros((data.ncells, data.nlev)),
    allocator=backend,
)
mask_out = gtx.as_field(
    (
        dims.CellDim,
        dims.KDim,
    ),
    data.mask_out,
    allocator=backend,
)

ksize = data.dz.shape[0]

for _x in range(int(args.itime) + 1):
    if _x == 1:  # Only start timing second iteration
        start_time = time.time()

    graupel_run = graupel_run.with_backend(backend)
    graupel_run(
        last_lev=ksize - 1,
        dz=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.dz[:, :]),
            allocator=backend,
        ),
        te=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.t[0, :, :]),
            allocator=backend,
        ),
        p=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.p[0, :, :]),
            allocator=backend,
        ),
        rho=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.rho[0, :, :]),
            allocator=backend,
        ),
        qve=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qv[0, :, :]),
            allocator=backend,
        ),
        qce=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qc[0, :, :]),
            allocator=backend,
        ),
        qre=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qr[0, :, :]),
            allocator=backend,
        ),
        qse=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qs[0, :, :]),
            allocator=backend,
        ),
        qie=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qi[0, :, :]),
            allocator=backend,
        ),
        qge=gtx.as_field(
            (
                dims.CellDim,
                dims.KDim,
            ),
            np.transpose(data.qg[0, :, :]),
            allocator=backend,
        ),
        dt=args.dt,
        qnc=args.qnc,
        t_out=t_out,
        qv_out=qv_out,
        qc_out=qc_out,
        qr_out=qr_out,
        qs_out=qs_out,
        qi_out=qi_out,
        qg_out=qg_out,
        pflx=pflx_out,
        pr=pr_out,
        ps=ps_out,
        pi=pi_out,
        pg=pg_out,
        pre=pre_out,
        offset_provider={"Koff": dims.KDim},
    )
    if _x == int(args.itime):  # End timer on last iteration
        end_time = time.time()

elapsed_time = end_time - start_time
print("For", int(args.itime), "iterations it took", elapsed_time, "seconds!")

data.prr_gsp = np.transpose(pr_out[dims.KDim(ksize - 1)].asnumpy())
data.prs_gsp = np.transpose(ps_out[dims.KDim(ksize - 1)].asnumpy())
data.pri_gsp = np.transpose(pi_out[dims.KDim(ksize - 1)].asnumpy())
data.prg_gsp = np.transpose(pg_out[dims.KDim(ksize - 1)].asnumpy())
data.pre_gsp = np.transpose(pre_out[dims.KDim(ksize - 1)].asnumpy())

write_fields(
    args.output_file,
    data.ncells,
    data.nlev,
    t=np.transpose(t_out.asnumpy()),
    qv=np.transpose(qv_out.asnumpy()),
    qc=np.transpose(qc_out.asnumpy()),
    qi=np.transpose(qi_out.asnumpy()),
    qr=np.transpose(qr_out.asnumpy()),
    qs=np.transpose(qs_out.asnumpy()),
    qg=np.transpose(qg_out.asnumpy()),
    prr_gsp=data.prr_gsp,
    pri_gsp=data.pri_gsp,
    prs_gsp=data.prs_gsp,
    prg_gsp=data.prg_gsp,
    pflx=np.transpose(pflx_out.asnumpy()),
    pre_gsp=data.pre_gsp,
)
