# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import os

import click
import numcodecs_observers
import xarray as xr
from data_compression_cscs_exclaim import utils
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver


@click.group()
def cli():
    pass


@cli.command("linear_quantization_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def linear_quantization_zlib_compressors(
    netcdf_file: str, field_to_compress: str, parameters_file: str
):
    from numcodecs_wasm_linear_quantize import LinearQuantize
    from numcodecs_wasm_zlib import Zlib

    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    linear_quantization_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    ds_linquant = {}
    metrics_total_linquant = {}

    for name, da in ds.items():
        if name != field_to_compress:
            continue

        linquant_metrics = dict(
            nbytes=BytesizeObserver(),
            instructions=WasmCodecInstructionCounterObserver(),
            timings=WalltimeObserver(),
        )

        linquant_compressor = CodecStack(
            LinearQuantize(bits=linear_quantization_bits, dtype=str(da.dtype)),
            Zlib(level=zlib_level),
        )

        with numcodecs_observers.observe(
            linquant_compressor,
            observers=linquant_metrics.values(),
        ) as linquant_compressor_:
            ds_linquant[name] = linquant_compressor_.encode_decode_data_array(da).compute()

        print(f"{da.long_name}" + ":")
        linquant_metrics = utils.format_compression_metrics(linquant_compressor, **linquant_metrics)
        print(linquant_metrics.to_string(index=False))

        metrics_total_linquant[name] = linquant_metrics.loc["Summary"]

    # Save compressed file
    output_dir = "./output_netCDF_files"
    filename = "ds_linquant.nc"
    output_path = os.path.join(output_dir, filename)
    ds = xr.Dataset(ds_linquant)
    ds.to_netcdf(output_path)


@cli.command("bitround_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def bitround_zlib_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    from numcodecs_wasm_bit_round import BitRound
    from numcodecs_wasm_zlib import Zlib

    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    bitround_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    ds_bitround = {}
    metrics_total_bitround = {}

    for name, da in ds.items():
        if name != field_to_compress:
            continue

        bitround_metrics = dict(
            nbytes=BytesizeObserver(),
            instructions=WasmCodecInstructionCounterObserver(),
            timings=WalltimeObserver(),
        )

        bitround_compressor = CodecStack(
            BitRound(keepbits=bitround_bits),
            Zlib(level=zlib_level),
        )

        with numcodecs_observers.observe(
            bitround_compressor,
            observers=bitround_metrics.values(),
        ) as bitround_compressor_:
            ds_bitround[name] = bitround_compressor_.encode_decode_data_array(da).compute()

        print(f"{da.long_name}" + ":")
        bitround_metrics = utils.format_compression_metrics(bitround_compressor, **bitround_metrics)
        print(bitround_metrics.to_string(index=False))

        metrics_total_bitround[name] = bitround_metrics.loc["Summary"]

    # Save compressed file
    output_dir = "./output_netCDF_files"
    filename = "ds_bitround.nc"
    output_path = os.path.join(output_dir, filename)
    ds = xr.Dataset(ds_bitround)
    ds.to_netcdf(output_path)


@cli.command("zfp_asinh_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def zfp_asinh_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    from numcodecs_wasm_asinh import Asinh
    from numcodecs_wasm_zfp import Zfp

    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    ds_zfp = {}
    metrics_total_zfp = {}

    for name, da in ds.items():
        if name != field_to_compress:
            continue

        zfp_metrics = dict(
            nbytes=BytesizeObserver(),
            instructions=WasmCodecInstructionCounterObserver(),
            timings=WalltimeObserver(),
        )

        zfp_compressor = CodecStack(
            Asinh(linear_width=asinh_linear_width),
            Zfp(mode=zfp_mode, tolerance=zfp_tolerance),
        )

        with numcodecs_observers.observe(
            zfp_compressor,
            observers=zfp_metrics.values(),
        ) as zfp_compressor_:
            ds_zfp[name] = zfp_compressor_.encode_decode_data_array(da).compute()

        print(f"{da.long_name}" + ":")
        zfp_metrics = utils.format_compression_metrics(zfp_compressor, **zfp_metrics)
        print(zfp_metrics.to_string(index=False))

        metrics_total_zfp[name] = zfp_metrics.loc["Summary"]

    # Save compressed file
    output_dir = "./output_netCDF_files"
    filename = "ds_zfp.nc"
    output_path = os.path.join(output_dir, filename)
    ds = xr.Dataset(ds_zfp)
    ds.to_netcdf(output_path)


@cli.command("sz3_eb_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def sz3_eb_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    from numcodecs_wasm_sz3 import Sz3

    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    ds_sz3 = {}
    metrics_total_sz3 = {}

    for name, da in ds.items():
        if name != field_to_compress:
            continue

        sz3_metrics = dict(
            nbytes=BytesizeObserver(),
            instructions=WasmCodecInstructionCounterObserver(),
            timings=WalltimeObserver(),
        )

        sz3_compressor = CodecStack(Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel))

        with numcodecs_observers.observe(
            sz3_compressor,
            observers=sz3_metrics.values(),
        ) as sz3_compressor_:
            ds_sz3[name] = sz3_compressor_.encode_decode_data_array(da).compute()

        print(f"{da.long_name}" + ":")
        sz3_metrics = utils.format_compression_metrics(sz3_compressor, **sz3_metrics)
        print(sz3_metrics.to_string(index=False))

        metrics_total_sz3[name] = sz3_metrics.loc["Summary"]

    # Save compressed file
    output_dir = "./output_netCDF_files"
    filename = "ds_sz3.nc"
    output_path = os.path.join(output_dir, filename)
    ds = xr.Dataset(ds_sz3)
    ds.to_netcdf(output_path)


@cli.command("help")
@click.pass_context
def help(ctx):
    for command in cli.commands.values():
        if command.name == "help":
            continue
        click.echo("-" * 80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as ctx:
            click.echo(command.get_help(ctx=ctx))
        click.echo()


if __name__ == "__main__":
    cli()
