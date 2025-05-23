import click
import inspect

import numcodecs_observers
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver
from numcodecs_combinators.stack import CodecStack

from data_compression_cscs_exclaim import utils


@click.group()
def cli():
    pass


@cli.command("linear_quantization_zlib_compressors")
@click.argument("netcdf_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("field_to_compress")
@click.argument("parameters_file", type=click.Path(exists=True, dir_okay=False))
def linear_quantization_zlib_compressors(netcdf_file: str, field_to_compress: str, parameters_file: str):
    from numcodecs_wasm_linear_quantize import LinearQuantize
    from numcodecs_wasm_zlib import Zlib
        
    ds = utils.open_netcdf(netcdf_file, field_to_compress)
    
    linear_quantization_bits, zlib_level = utils.get_filter_parameters(parameters_file, inspect.currentframe().f_code.co_name)
    
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
        print(linquant_metrics)

        metrics_total_linquant[name] = linquant_metrics.loc["Summary"]


@cli.command("help")
@click.pass_context
def help(ctx):
    for command in cli.commands.values():
        if command.name == "help":
            continue
        click.echo("-"*80)
        click.echo()
        with click.Context(command, parent=ctx.parent, info_name=command.name) as ctx:
            click.echo(command.get_help(ctx=ctx))
        click.echo()


if __name__ == '__main__':
    cli()
