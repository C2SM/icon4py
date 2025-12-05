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
import pathlib
import time

from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, utils
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


def setup_graupel(
    inp: common.GraupelInput, dt: float, qnc: float, backend: model_backends.BackendLike
):
    with utils.recursion_limit(10**4):  # TODO(havogt): make an option in gt4py?
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
            },
            offset_provider={"Koff": dims.KDim},
        )
        gtx.wait_for_compilation()
        return graupel_run_program


def main():
    args = get_args()

    backend = model_backends.BACKENDS[args.backend]
    allocator = model_backends.get_allocator(backend)

    inp = common.GraupelInput.load(filename=pathlib.Path(args.input_file), allocator=allocator)
    out = common.GraupelOutput.allocate(
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}), allocator=allocator
    )

    graupel_run_program = setup_graupel(inp, dt=args.dt, qnc=args.qnc, backend=backend)

    start_time = None
    for _x in range(int(args.itime) + 1):
        if _x == 1:  # Only start timing second iteration
            device_utils.sync(backend)
            start_time = time.time()

        graupel_run_program(
            dz=inp.dz,
            te=inp.t,
            p=inp.p,
            rho=inp.rho,
            q_in=inp.q,
            t_out=out.t,
            q_out=out.q,
            pflx=out.pflx,
            pr=out.pr,
            ps=out.ps,
            pi=out.pi,
            pg=out.pg,
            pre=out.pre,
        )
    device_utils.sync(backend)
    end_time = time.time()

    if start_time is not None:
        elapsed_time = end_time - start_time
        print("For", int(args.itime), "iterations it took", elapsed_time, "seconds!")

    out.write(args.output_file)


if __name__ == "__main__":
    main()
