# Data Compression Project

TODO

## Installation

```commandline
git clone -b data_compression https://github.com/C2SM/icon4py.git
cd ./icon4py/tools/src/icon4py/tools/data_compression/prototyping
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

```
--------------------------------------------------------------------------------

Usage: data_compression_cscs_exclaim linear_quantization_zlib_compressors
           [OPTIONS] NETCDF_FILE FIELD_TO_COMPRESS PARAMETERS_FILE

Example:
```

data_compression_cscs_exclaim linear_quantization_zlib_compressors netCDF_files/tigge_pl_t_q_dx=2_2024_08_02.nc t parameters.yaml

````

Options:
  --help  Show this message and exit.

--------------------------------------------------------------------------------

```
````

## UI for compressor pre-eval

model_predict_ui.py can be used to evaluate which compressor works best.

It is possible to tweak compressor parameters on-the-fly.

```
prototyping % streamlit run ./src/data_compression_cscs_exclaim/model_predict_ui.py [OPTIONAL] --server.maxUploadSize=FILE_SIZE_MB

```
