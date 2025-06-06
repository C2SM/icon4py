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
import numpy as np
import pywt
import xarray as xr
from data_compression_cscs_exclaim import utils
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zlib import Zlib


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


@cli.command("models_evaluation")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def models_evaluation(netcdf_file: str, field_to_compress: str, parameters_file: str):
    # Array slicing
    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    ds_coords = dict(ds.coords).keys()
    lat_key = [key for key in ds_coords if key.startswith("lat")][0]
    lon_key = [key for key in ds_coords if key.startswith("lon")][0]
    lat_upper = round(ds[field_to_compress][lat_key].shape[0] * 0.2)
    lon_upper = round(ds[field_to_compress][lon_key].shape[0] * 0.2)
    ds = ds.isel(latitude=slice(0, lat_upper), longitude=slice(0, lon_upper))
    dwt_dists = {}

    # linear_quantization_zlib_compressors
    linear_quantization_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, "linear_quantization_zlib_compressors"
    )
    ds_linquant = {}
    linquant_compressor = CodecStack(
        LinearQuantize(bits=linear_quantization_bits, dtype=str(ds[field_to_compress].dtype)),
        Zlib(level=zlib_level),
    )
    ds_linquant[field_to_compress] = linquant_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    lquant_dwt_distances = utils.calc_dwt_dist(
        ds_linquant[field_to_compress], ds[field_to_compress], n_levels=4
    )
    dwt_dists["lquant_dwt_distances"] = lquant_dwt_distances
    print("Linear Quantization DWT disctance: " + f"{lquant_dwt_distances}")

    # bitround_zlib_compressors
    bitround_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, "bitround_zlib_compressors"
    )
    ds_bitround = {}
    bitround_compressor = CodecStack(
        BitRound(keepbits=bitround_bits),
        Zlib(level=zlib_level),
    )
    ds_bitround[field_to_compress] = bitround_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    bitround_dwt_distances = utils.calc_dwt_dist(
        ds_bitround[field_to_compress], ds[field_to_compress], n_levels=4
    )
    dwt_dists["bitround_dwt_distances"] = bitround_dwt_distances
    print("Bit Round DWT disctance: " + f"{bitround_dwt_distances}")

    # zfp_asinh_compressors
    asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
        parameters_file, "zfp_asinh_compressors"
    )
    ds_zfp = {}
    zfp_compressor = CodecStack(
        Asinh(linear_width=asinh_linear_width),
        Zfp(mode=zfp_mode, tolerance=zfp_tolerance),
    )
    ds_zfp[field_to_compress] = zfp_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    zfp_dwt_distances = utils.calc_dwt_dist(ds_zfp[field_to_compress], ds[field_to_compress], n_levels=4)
    dwt_dists["zfp_dwt_distances"] = zfp_dwt_distances
    print("Zfp DWT disctance: " + f"{zfp_dwt_distances}")

    # sz3_eb_compressors
    sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(parameters_file, "sz3_eb_compressors")
    ds_sz3 = {}
    sz3_compressor = CodecStack(Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel))
    ds_sz3[field_to_compress] = sz3_compressor.encode_decode_data_array(
        ds[field_to_compress]
    ).compute()
    sz3_dwt_distances = utils.calc_dwt_dist(ds_sz3[field_to_compress], ds[field_to_compress], n_levels=4)
    dwt_dists["sz3_eb_compressors"] = sz3_dwt_distances
    print("SZ3 DWT disctance: " + f"{sz3_dwt_distances}")

    min_key = min(dwt_dists, key=dwt_dists.get)
    print("Best compression method: " + f"{min_key}")


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
