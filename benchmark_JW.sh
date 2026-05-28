#!/bin/bash
#SBATCH --job-name=JW4Py_benchmark
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --uenv=icon/25.2:v4
#SBATCH -A csstaff
#SBATCH --view=default

# Go to the root of the icon4py repository to run the script from there
ICON4PY_GIT_ROOT=$(git rev-parse --show-toplevel)
cd $ICON4PY_GIT_ROOT

source .venv/bin/activate

export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export PYTHONOPTIMIZE=2
# export GT4PY_COLLECT_METRICS_LEVEL=10
# export GT4PY_ADD_GPU_TRACE_MARKERS="1"
# export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"

export ICON_GRID="icon_grid_0013_R02B04_R.nc" #"icon_grid_0021_R02B06_G.nc" #"icon_grid_0013_R02B04_R.nc" #"icon_grid_0023_R02B07_G.nc"
SUFFIX=""
if [[ "$ICON_GRID" == *"R02B04"* ]]; then
    SUFFIX="R02B04"
elif [[ "$ICON_GRID" == *"R02B06"* ]]; then
    SUFFIX="R02B06"
elif [[ "$ICON_GRID" == *"R02B07"* ]]; then
    SUFFIX="R02B07"
fi
SUFFIX="${SUFFIX}_latest"
export GT4PY_BUILD_CACHE_DIR="amd_profiling_JW_${SUFFIX}"

export LD_LIBRARY_PATH=$(pwd):${LD_LIBRARY_PATH}
# /capstor/scratch/cscs/ioannmag/NSYS_20262/nsight_systems_2026_2/target-linux-sbsa-armv8/nsys profile --trace=cuda,nvtx,osrt \
python model/standalone_driver/src/icon4py/model/standalone_driver/main.py --grid-file-path $(realpath ${ICON_GRID}) --icon4py-backend dace_gpu --log-level warning --output-path $(pwd)/standalone_driver_output_${SUFFIX}
