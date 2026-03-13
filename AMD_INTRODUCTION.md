# Icon4py performance on MI300

## Quickstart

```
# Connect to Beverin (CSCS system with MI300A)
ssh beverin.cscs.ch
```

In Beverin:

```
# Enter scratch directory
cd $SCRATCH

# Clone icon4py and checkout the correct branch
git clone git@github.com:C2SM/icon4py.git
cd icon4py
git checkout amd_profiling

# Pull the correct `uenv` image. *!* NECESSARY ONLY ONCE *!*
uenv image pull build::prgenv-gnu/25.12:2333839235

# Start the uenv and mount the ROCm 7.1.0 environment. *!* This needs to be executed before running anything everytime *!*
uenv start --view default prgenv-gnu/25.12:2333839235

# Install the necessary venv
bash amd_scripts/install_icon4py_venv.sh

# Source venv
source .venv/bin/activate

# Source other necessary environment variables
source amd_scripts/setup_env.sh

# Set GT4Py related environment variables
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export GT4PY_BUILD_CACHE_LIFETIME=persistent
export GT4PY_BUILD_CACHE_DIR=amd_profiling_granule
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_DYCORE_ENABLE_METRICS="1"
export GT4PY_ADD_GPU_TRACE_MARKERS="1"
export HIPFLAGS="-std=c++17 -fPIC -O3 -march=native -Wno-unused-parameter -save-temps -Rpass-analysis=kernel-resource-usage"
```
