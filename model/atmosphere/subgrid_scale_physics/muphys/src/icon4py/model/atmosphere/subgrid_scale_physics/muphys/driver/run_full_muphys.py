#!/usr/bin/env python
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib
import time

import netCDF4
import numpy as np
from gt4py import next as gtx
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import utils
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import saturation_adjustment
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import graupel
from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.utils import device_utils


# TODO(havogt): make similar to icon4py driver structure


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", metavar="output_file", dest="output_file", help="output filename", default="output.nc"
    )
    parser.add_argument(
        "-b", metavar="backend", dest="backend", help="gt4py backend", default="gtfn_cpu"
    )
    parser.add_argument("input_file", help="input data file")
    parser.add_argument("itime", help="time-index", nargs="?", default=0)
    parser.add_argument("dt", help="timestep", nargs="?", default=30.0)
    parser.add_argument("qnc", help="Water number concentration", nargs="?", default=100.0)

    return parser.parse_args()


# TODO double check the sizes of pxxx vars (should be nlev+1?)


def _calc_dz(z: np.ndarray) -> np.ndarray:
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def _as_field_from_nc(
    dataset: netCDF4.Dataset,
    allocator: gtx_typing.FieldBufferAllocationUtil,
    varname: str,
    optional: bool = False,
    dtype: np.dtype | None = None,
) -> gtx.Field[dims.CellDim, dims.KDim] | None:
    if optional and varname not in dataset.variables:
        return None

    var = dataset.variables[varname]
    if var.dimensions[0] == "time":
        var = var[0, :, :]
    data = np.transpose(var)
    if dtype is not None:
        data = data.astype(dtype)
    return gtx.as_field(
        (dims.CellDim, dims.KDim),
        data,
        allocator=allocator,
    )


def _field_to_nc(
    dataset: netCDF4.Dataset,
    dims: tuple[str, str],
    varname: str,
    field: gtx.Field[dims.CellDim, dims.KDim],
    dtype: np.dtype = np.float64,
) -> None:
    var = dataset.createVariable(varname, dtype, dims)
    var[...] = field.asnumpy().transpose()


@dataclasses.dataclass
class GraupelInput:
    ncells: int
    nlev: int
    dz: gtx.Field[dims.CellDim, dims.KDim]
    p: gtx.Field[dims.CellDim, dims.KDim]
    rho: gtx.Field[dims.CellDim, dims.KDim]
    t: gtx.Field[dims.CellDim, dims.KDim]
    qv: gtx.Field[dims.CellDim, dims.KDim]
    qc: gtx.Field[dims.CellDim, dims.KDim]
    qi: gtx.Field[dims.CellDim, dims.KDim]
    qr: gtx.Field[dims.CellDim, dims.KDim]
    qs: gtx.Field[dims.CellDim, dims.KDim]
    qg: gtx.Field[dims.CellDim, dims.KDim]

    @classmethod
    def load(
        cls, filename: pathlib.Path | str, allocator: gtx_typing.FieldBufferAllocationUtil
    ) -> None:
        with netCDF4.Dataset(filename, mode="r") as ncfile:
            try:
                ncells = len(ncfile.dimensions["cell"])
            except KeyError:
                ncells = len(ncfile.dimensions["ncells"])

            nlev = len(ncfile.dimensions["height"])

            dz = _calc_dz(ncfile.variables["zg"])

            field_from_nc = functools.partial(
                _as_field_from_nc, ncfile, allocator, dtype=np.float64
            )
            return cls(
                ncells=ncells,
                nlev=nlev,
                dz=gtx.as_field((dims.CellDim, dims.KDim), np.transpose(dz), allocator=allocator),
                t=field_from_nc("ta"),
                p=field_from_nc("pfull"),
                qs=field_from_nc("qs"),
                qi=field_from_nc("cli"),
                qg=field_from_nc("qg"),
                qv=field_from_nc("hus"),
                qc=field_from_nc("clw"),
                qr=field_from_nc("qr"),
                rho=field_from_nc("rho"),
            )


@dataclasses.dataclass
class GraupelOutput:
    t: gtx.Field[dims.CellDim, dims.KDim]
    qv: gtx.Field[dims.CellDim, dims.KDim]
    qc: gtx.Field[dims.CellDim, dims.KDim]
    qi: gtx.Field[dims.CellDim, dims.KDim]
    qr: gtx.Field[dims.CellDim, dims.KDim]
    qs: gtx.Field[dims.CellDim, dims.KDim]
    qg: gtx.Field[dims.CellDim, dims.KDim]
    t_tmp: gtx.Field[dims.CellDim, dims.KDim]
    qv_tmp: gtx.Field[dims.CellDim, dims.KDim]
    qc_tmp: gtx.Field[dims.CellDim, dims.KDim]
    pflx: gtx.Field[dims.CellDim, dims.KDim] | None
    pr: gtx.Field[dims.CellDim, dims.KDim] | None
    ps: gtx.Field[dims.CellDim, dims.KDim] | None
    pi: gtx.Field[dims.CellDim, dims.KDim] | None
    pg: gtx.Field[dims.CellDim, dims.KDim] | None
    pre: gtx.Field[dims.CellDim, dims.KDim] | None

    @classmethod
    def allocate(cls, allocator: gtx_typing.FieldBufferAllocationUtil, domain: gtx.Domain):
        zeros = functools.partial(gtx.zeros, domain=domain, allocator=allocator)
        # TODO +1 size fields?
        return cls(**{field.name: zeros() for field in dataclasses.fields(cls)})

    @classmethod
    def load(cls, filename: pathlib.Path | str, allocator: gtx_typing.FieldBufferAllocationUtil):
        with netCDF4.Dataset(filename, mode="r") as ncfile:
            field_from_nc = functools.partial(_as_field_from_nc, ncfile, allocator)
            return cls(
                t=field_from_nc("ta"),
                qv=field_from_nc("hus"),
                qc=field_from_nc("clw"),
                qi=field_from_nc("cli"),
                qr=field_from_nc("qr"),
                qs=field_from_nc("qs"),
                qg=field_from_nc("qg"),
                t_tmp=field_from_nc("ta"),
                qv_tmp=field_from_nc("hus"),
                qc_tmp=field_from_nc("clw"),
                pflx=field_from_nc("pflx", optional=True),
                pr=field_from_nc("prr_gsp", optional=True),
                ps=field_from_nc("prs_gsp", optional=True),
                pi=field_from_nc("pri_gsp", optional=True),
                pg=field_from_nc("prg_gsp", optional=True),
                pre=field_from_nc("pre_gsp", optional=True),
            )

    def write(self, filename: pathlib.Path | str):
        ncells = self.t.shape[0]
        nlev = self.t.shape[1]

        with netCDF4.Dataset(filename, mode="w") as ncfile:
            ncfile.createDimension("ncells", ncells)
            ncfile.createDimension("height", nlev)
            ncfile.createDimension("height1", nlev + 1)  # what's the reason for the +1 fields here?

            write_height_field = functools.partial(
                _field_to_nc, ncfile, ("height", "ncells"), dtype=np.float64
            )
            write_height1_field = functools.partial(  # TODO
                _field_to_nc, ncfile, ("height1", "ncells"), dtype=np.float64
            )

            write_height_field("ta", self.t)
            write_height_field("hus", self.qv)
            write_height_field("clw", self.qc)
            write_height_field("cli", self.qi)
            write_height_field("qr", self.qr)
            write_height_field("qs", self.qs)
            write_height_field("qg", self.qg)
            if self.pflx is not None:
                write_height_field("pflx", self.pflx)
            if self.pr is not None:
                write_height_field("prr_gsp", self.pr)  # TODO height1?
            if self.ps is not None:
                write_height_field("prs_gsp", self.ps)  # TODO
            if self.pi is not None:
                write_height_field("pri_gsp", self.pi)  # TODO
            if self.pg is not None:
                write_height_field("prg_gsp", self.pg)  # TODO
            if self.pre is not None:
                write_height_field("pre_gsp", self.pre)  # TODO


def setup_saturation_adjustment(inp: GraupelInput, dt: float, qnc: float, backend: model_backends.BackendLike):
    with utils.recursion_limit(10**4):  # TODO thread safe?
        saturation_adjustment_run_program = model_options.setup_program(
            backend=backend,
            program=saturation_adjustment.saturation_adjustment_run,
            constant_args={"dt": dt, "qnc": qnc},
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": inp.ncells,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(inp.nlev),
            },
            offset_provider={"Koff": dims.KDim},
        )
        gtx.wait_for_compilation()
        return saturation_adjustment_run_program


def setup_graupel(inp: GraupelInput, dt: float, qnc: float, backend: model_backends.BackendLike):
    with utils.recursion_limit(10**4):  # TODO thread safe?
        graupel_run_program = model_options.setup_program(
            backend=backend,
            program=graupel.graupel_run,
            constant_args={"dt": dt, "qnc": qnc},
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": inp.ncells,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(inp.nlev),
                "last_lev": gtx.int32(inp.nlev - 1),
            },
            offset_provider={"Koff": dims.KDim},
        )
        gtx.wait_for_compilation()
        return graupel_run_program


def main():
    args = get_args()

    backend = model_backends.BACKENDS[args.backend]
    allocator = model_backends.get_allocator(backend)

    inp = GraupelInput.load(filename=pathlib.Path(args.input_file), allocator=allocator)
    out = GraupelOutput.allocate(
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}), allocator=allocator
    )

    saturation_adjustment_run_program = setup_saturation_adjustment(inp, dt=args.dt, qnc=args.qnc, backend=backend)
    graupel_run_program = setup_graupel(inp, dt=args.dt, qnc=args.qnc, backend=backend)

    start_time = None
    for _x in range(int(args.itime) + 1):
        if _x == 1:  # Only start timing second iteration
            device_utils.sync(backend)
            start_time = time.time()

        saturation_adjustment_program(
            te=inp.t,
            qve=inp.qve,
            qce=inp.qce,
            qre=inp.qre,
            qse=inp.qse,
            qie=inp.qie,
            qge=inp.qge,
            rho=inp.rho,
            te_out=out.t,  # Temperature
            qve_out=out.qv,  # Specific humidity
            qce_out=out.qc,  # Specific cloud water content
        )

        graupel_run_program(
            dz=inp.dz,
            te=out.t,        # output of sat_adj
            p=inp.p,
            rho=inp.rho,
            qve=out.qv,      # output of sat_adj
            qce=out.qc,      # output of sat_adj
            qre=inp.qr,
            qse=inp.qs,
            qie=inp.qi,
            qge=inp.qg,
            t_out=out.t_tmp,
            qv_out=out.qv_tmp,
            qc_out=out.qc_tmp,
            qr_out=out.qr,
            qs_out=out.qs,
            qi_out=out.qi,
            qg_out=out.qg,
            pflx=out.pflx,
            pr=out.pr,
            ps=out.ps,
            pi=out.pi,
            pg=out.pg,
            pre=out.pre,
        )

        saturation_adjustment_program(
            te=out.t_tmp,    # output of graupel
            qve=out.qv_tmp,  # output of graupel
            qce=out.qc_tmp,  # output of graupel
            qre=out.qr,
            qse=out.qs,
            qie=out.qi,
            qge=out.qg,
            rho=inp.rho,
            te_out=out.t,    # Temperature
            qve_out=out.qv,  # Specific humidity
            qce_out=out.qc,  # Specific cloud water content
        )

    device_utils.sync(backend)
    end_time = time.time()

    if start_time is not None:
        elapsed_time = end_time - start_time
        print("For", int(args.itime), "iterations it took", elapsed_time, "seconds!")

    out.write(args.output_file)


if __name__ == "__main__":
    main()
