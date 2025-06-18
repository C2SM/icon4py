# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import click
from tqdm import tqdm
import itertools
import numcodecs
import numcodecs.zarr3
from data_compression_cscs_exclaim import utils
from numcodecs_combinators.stack import CodecStack
from numcodecs_wasm_asinh import Asinh
from numcodecs_wasm_bit_round import BitRound
from numcodecs_wasm_linear_quantize import LinearQuantize
from numcodecs_wasm_sz3 import Sz3
from numcodecs_wasm_zfp import Zfp
from numcodecs_wasm_zlib import Zlib
from zarr_any_numcodecs import AnyNumcodecsArrayArrayCodec, AnyNumcodecsArrayBytesCodec, AnyNumcodecsBytesBytesCodec

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations",
    category=UserWarning,
    module="numcodecs.zarr3"
)


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

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(LinearQuantize(bits=linear_quantization_bits, dtype=str(ds[field_to_compress].dtype)))],
        compressors=[AnyNumcodecsBytesBytesCodec(Zlib(level=zlib_level))],
    )


@cli.command("bitround_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def bitround_zlib_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    bitround_bits, zlib_level = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(BitRound(keepbits=bitround_bits))],
        compressors=[AnyNumcodecsBytesBytesCodec(Zlib(level=zlib_level))],
    )


@cli.command("zfp_asinh_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def zfp_asinh_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    asinh_linear_width, zfp_mode, zfp_tolerance = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=[AnyNumcodecsArrayArrayCodec(Asinh(linear_width=asinh_linear_width))],
        compressors=None,
        serializer=numcodecs.zarr3.ZFPY(tolerance=zfp_tolerance)
    )


@cli.command("sz3_eb_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def sz3_eb_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ds = utils.open_netcdf(netcdf_file, field_to_compress)

    sz3_eb_mode, sz3_eb_rel = utils.get_filter_parameters(
        parameters_file, inspect.currentframe().f_code.co_name
    )

    sz3_ = Sz3(eb_mode=sz3_eb_mode, eb_rel=sz3_eb_rel)

    utils.compress_with_zarr(ds[field_to_compress], netcdf_file, field_to_compress,
        filters=None,
        compressors=None,
        serializer=AnyNumcodecsArrayBytesCodec(sz3_)
    )


@cli.command("summarize_compression")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def summarize_compression(netcdf_file: str, field_to_compress: str, parameters_file: str):
    ## https://numcodecs.readthedocs.io/en/stable/zarr3.html#zarr-3-codecs
    
    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    da = ds[field_to_compress]
    
    gather_all = {}

    compressors = utils.compressor_space(da)
    filters = utils.filter_space(da)
    serializers = utils.serializer_space(da)
    
    num_compressors = len(compressors)
    num_filters = len(filters)
    num_serializers = len(serializers)
    num_loops = num_compressors * num_filters * num_serializers
    click.echo(f"Number of loops: {num_loops} ({num_compressors} compressors, {num_filters} filters, {num_serializers} serializers)")
    
    for compressor, filter, serializer in tqdm(
        itertools.product(compressors, filters, serializers),
        total=num_loops,
        desc="Executing compression combinations",
    ):
        d = utils.compress_with_zarr(da, netcdf_file, field_to_compress,
            filters=[filter,],
            compressors=[compressor,],
            serializer=serializer,
            echo=False
        )


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
