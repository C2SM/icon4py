from collections.abc import Sequence
from collections import OrderedDict

import numpy as np
import click
import humanize
import xarray as xr
import sys
import yaml
import traceback


def open_netcdf(netcdf_file: str, field_to_compress: str):
    ds = xr.open_dataset(netcdf_file)
    
    if field_to_compress not in ds.data_vars:
        click.echo(f"Field {field_to_compress} not found in NetCDF file.")
        click.echo("Aborting...")
        sys.exit(1)
    
    click.echo(f"netcdf_file.nbytes = {humanize.naturalsize(ds.nbytes, binary=True)}")
    click.echo(f"field_to_compress.nbytes = {humanize.naturalsize(ds[field_to_compress].nbytes, binary=True)}")

    return ds


def ordered_yaml_loader():
    class OrderedLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return OrderedLoader


def get_filter_parameters(parameters_file: str, filter_name: str):
    with open(parameters_file, 'r') as f:
        params = yaml.load(f, Loader=ordered_yaml_loader())
        
    try:
        filter_config = params[filter_name]["params"]
        return tuple(filter_config.values())
    
    except Exception as e:
        click.echo("An unexpected error occurred:", err=True)
        traceback.print_exc(file=sys.stderr)


def compute_relative_errors(da_compressed, da):
    da_error = da_compressed - da

    norm_L1_error = np.abs(da_error).sum().values
    norm_L2_error = np.sqrt((da_error**2).sum().values)
    norm_Linf_error = np.abs(da_error).max().values

    norm_L1_original = np.abs(da).sum().values
    norm_L2_original = np.sqrt((da**2).sum().values)
    norm_Linf_original = np.abs(da).max().values

    relative_error_L1 = norm_L1_error / norm_L1_original
    relative_error_L2 = norm_L2_error / norm_L2_original
    relative_error_Linf = norm_Linf_error / norm_Linf_original

    return {
        "Relative_Error_L1": relative_error_L1,
        "Relative_Error_L2": relative_error_L2,
        "Relative_Error_Linf": relative_error_Linf,
    }


def format_compression_metrics(
    codecs: Sequence["numcodecs.abc.Codec"],
    *,
    nbytes: "numcodecs_observers.bytesize.BytesizeObserver",
    instructions: "None | numcodecs_wasm.WasmCodecInstructionCounterObserver" = None,
    timings: "None | numcodecs_observers.walltime.WalltimeObserver" = None,
):
    import pandas as pd
    from numcodecs_observers.hash import HashableCodec

    codecs = tuple(codecs)

    encoded_bytes = {c: sum(e.post for e in es) for c, es in nbytes.encode_sizes.items()}
    decoded_bytes = {c: sum(d.post for d in ds) for c, ds in nbytes.decode_sizes.items()}

    table = pd.DataFrame(
        {
            "Codec": [str(c) for c in codecs] + ["Summary"],
            "compression ratio [raw B / enc B]": [
                round(decoded_bytes[HashableCodec(c)] / encoded_bytes[HashableCodec(c)], 2)
                for c in codecs
            ]
            + (
                [
                    round(
                        decoded_bytes[HashableCodec(codecs[0])]
                        / encoded_bytes[HashableCodec(codecs[-1])],
                        2,
                    )
                ]
                if len(codecs) > 0
                else [1.0]
            ),
        }
    ).set_index(["Codec"])

    if instructions is not None:
        table["encode instructions [#/B]"] = [
            (
                round(
                    sum(instructions.encode_instructions[HashableCodec(c)])
                    / decoded_bytes[HashableCodec(c)],
                    1,
                )
                if HashableCodec(c) in instructions.encode_instructions
                else "<unknown>"
            )
            for c in codecs
        ] + (
            [
                round(
                    sum(sum(instructions.encode_instructions[HashableCodec(c)]) for c in codecs)
                    / decoded_bytes[HashableCodec(codecs[0])],
                    1,
                )
                if all(HashableCodec(c) in instructions.encode_instructions for c in codecs)
                else "<unknown>"
            ]
            if len(codecs) > 0
            else [0.0]
        )

        table["decode instructions [#/B]"] = [
            (
                round(
                    sum(instructions.decode_instructions[HashableCodec(c)])
                    / decoded_bytes[HashableCodec(c)],
                    1,
                )
                if HashableCodec(c) in instructions.decode_instructions
                else "<unknown>"
            )
            for c in codecs
        ] + (
            [
                round(
                    sum(sum(instructions.decode_instructions[HashableCodec(c)]) for c in codecs)
                    / decoded_bytes[HashableCodec(codecs[0])],
                    1,
                )
                if all(HashableCodec(c) in instructions.decode_instructions for c in codecs)
                else "<unknown>"
            ]
            if len(codecs) > 0
            else [0.0]
        )

    if timings is not None:
        table["encode throughput [raw GB/s]"] = [
            round(
                1e-9
                * decoded_bytes[HashableCodec(c)]
                / sum(timings.encode_times[HashableCodec(c)]),
                2,
            )
            for c in codecs
        ] + (
            [
                round(
                    1e-9
                    * decoded_bytes[HashableCodec(codecs[0])]
                    / sum(sum(timings.encode_times[HashableCodec(c)]) for c in codecs),
                    2,
                )
            ]
            if len(codecs) > 0
            else [0.0]
        )

        table["decode throughput [raw GB/s]"] = [
            round(
                1e-9
                * decoded_bytes[HashableCodec(c)]
                / sum(timings.decode_times[HashableCodec(c)]),
                2,
            )
            for c in codecs
        ] + (
            [
                round(
                    1e-9
                    * decoded_bytes[HashableCodec(codecs[0])]
                    / sum(sum(timings.decode_times[HashableCodec(c)]) for c in codecs),
                    2,
                )
            ]
            if len(codecs) > 0
            else [0.0]
        )

    return table
