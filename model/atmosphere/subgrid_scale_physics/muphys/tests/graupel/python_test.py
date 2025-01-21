#!/usr/bin/env python
import sys
import netCDF4
import numpy as np
import argparse
import pdb

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import graupel_run
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat

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
    parser.add_argument("qnc", help="TODO:qnc-descripion", nargs="?", default=100.0)
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
        self.prr_gsp = np.zeros(self.ncells, np.float64)
        self.pri_gsp = np.zeros(self.ncells, np.float64)
        self.prs_gsp = np.zeros(self.ncells, np.float64)
        self.prg_gsp = np.zeros(self.ncells, np.float64)
        self.pre_gsp = np.zeros(self.ncells, np.float64)
        self.pflx = np.zeros((self.nlev, self.ncells), np.float64)
        # allocate dz:
        self.dz = np.zeros(self.ncells * self.nlev, np.float64)
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
#     qi=data.qi,
#     qr=data.qr,
#     qs=data.qs,
#     qg=data.qg,
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
    t=data.t,
    qv=data.qv,
    qc=data.qc,
    qi=data.qi,
    qr=data.qr,
    qs=data.qs,
    qg=data.qg,
    prr_gsp=data.prr_gsp,
    pri_gsp=data.pri_gsp,
    prs_gsp=data.prs_gsp,
    prg_gsp=data.prg_gsp,
    pflx=data.pflx,
    pre_gsp=data.pre_gsp,
)
