#!/usr/bin/env python
import sys
import netCDF4
import numpy as np
import argparse
import pdb
import gt4py.next as gtx


from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import graupel_run
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc

K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

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
        nc = netCDF4.Dataset(args.input_file)
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
        self.t_out     = np.zeros((self.ncells,self.nlev), np.float64)
        self.qv_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.qc_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.qi_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.qr_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.qs_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.qg_out    = np.zeros((self.ncells,self.nlev), np.float64)
        self.pflx_out  = np.zeros((self.ncells,self.nlev), np.float64)
        self.prr_gsp   = np.zeros(self.ncells, np.float64)
        self.pri_gsp   = np.zeros(self.ncells, np.float64)
        self.prs_gsp   = np.zeros(self.ncells, np.float64)
        self.prg_gsp   = np.zeros(self.ncells, np.float64)
        self.pre_gsp   = np.zeros(self.ncells, np.float64)
        # allocate dz:
        self.dz = np.zeros((self.ncells, self.nlev), np.float64)
        # calc dz:
#        py_graupel.calc_dz(z=self.z, dz=self.dz, ncells=self.ncells, nlev=self.nlev)


def write_fields(
    output_filename,
    ncells,
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
    ncfile = netCDF4.Dataset(output_filename, mode="w")
    ncells_dim = ncfile.createDimension("ncells", ncells)
    height_dim = ncfile.createDimension("height", nlev)
    height1_dim = ncfile.createDimension("height1", 1)
    ta_var = ncfile.createVariable("ta", np.double, ("height", "ncells"))
    hus_var = ncfile.createVariable("hus", np.double, ("height", "ncells"))
    clw_var = ncfile.createVariable("clw", np.double, ("height", "ncells"))
    cli_var = ncfile.createVariable("cli", np.double, ("height", "ncells"))
    qr_var = ncfile.createVariable("qr", np.double, ("height", "ncells"))
    qs_var = ncfile.createVariable("qs", np.double, ("height", "ncells"))
    qg_var = ncfile.createVariable("qg", np.double, ("height", "ncells"))
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


args = get_args()

set_lib_path(args.ldir)
# import py_graupel


data = Data(args)

# grpl = py_graupel.Graupel()
# grpl.initialize()

# if args.with_sat_adj:
#     py_graupel.saturation_adjustment(
#         ncells=data.ncells,
#         nlev=data.nlev,
#         ta=data.t,
#         qv=data.qv,
#         qc=data.qc,
#         qr=data.qr,
#         total_ice=data.qg + data.qs + data.qi,
#         rho=data.rho,
#     )
ksize = data.dz.shape[1]
k = gtx.as_field( (dims.KDim, ), np.arange(0,ksize,dtype=np.int32) )
t_out = gtx.as_field((dims.CellDim, dims.KDim,), data.t_out)
qv_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qv_out)
qc_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qc_out)
qr_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qr_out)
qs_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qs_out)
qi_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qi_out)
qg_out = gtx.as_field((dims.CellDim, dims.KDim,), data.qg_out)
graupel_run( k = k,
             last_lev = ksize-1,
             dz  = gtx.as_field((dims.CellDim, dims.KDim,), data.dz),
             te  = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.t[0,:,:])),
             p   = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.p[0,:,:])),
             rho = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.rho[0,:,:])),
             qve = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qv[0,:,:])),
             qce = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qc[0,:,:])),
             qre = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qr[0,:,:])),
             qse = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qs[0,:,:])),
             qie = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qi[0,:,:])),
             qge = gtx.as_field((dims.CellDim, dims.KDim,), np.transpose(data.qg[0,:,:])),
             dt  = args.dt,
             qnc = args.qnc,
             t_out  = t_out,
             qv_out = qv_out,
             qc_out = qc_out,
             qr_out = qr_out,
             qs_out = qs_out,
             qi_out = qi_out,
             qg_out = qg_out,
             offset_provider={"Koff": K})

# grpl.run(
#     ncells=data.ncells,
#     nlev=data.nlev,
#     dt=args.dt,
#     dz=data.dz,
#     t=data.t,
#     rho=data.rho,
#     p=data.p,
#     qv=data.qv,
#     qc=data.qc,
#     qg=data.qg,
#     qi=data.qi,
#     qr=data.qr,
#     qs=data.qs,
#     qnc=args.qnc,
#     prr_gsp=data.prr_gsp,
#     pri_gsp=data.pri_gsp,
#     prs_gsp=data.prs_gsp,
#     prg_gsp=data.prg_gsp,
#     pflx=data.pflx,
#     pre_gsp=data.pre_gsp,
# )

# if args.with_sat_adj:
#     py_graupel.saturation_adjustment(
#         ncells=data.ncells,
#         nlev=data.nlev,
#         ta=data.t,
#         qv=data.qv,
#         qc=data.qc,
#         qr=data.qr,
#         total_ice=data.qg + data.qs + data.qi,
#         rho=data.rho,
#     )

# grpl.finalize()

write_fields(
    args.output_file,
    data.ncells,
    data.nlev,
    t= np.transpose(t_out.asnumpy()),
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
    pflx=np.transpose(data.pflx_out),
    pre_gsp=data.pre_gsp,
)
