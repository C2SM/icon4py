INITIAL_DIR=$(pwd)

# unset PYTHONPATH and set PYTHONUSERBASE to avoid conflicts
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"

# create and activate a new relocatable venv using uv
# in this case we explicitly select the python interpreter from the uenv view
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy /dev/shm/$USER/.venv
cd /dev/shm/$USER
source .venv/bin/activate

cd $INITIAL_DIR
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH="80;90"
export GHEX_TRANSPORT_BACKEND=MPI
export CC=$(which gcc)
export CXX=$(which g++)
export MPICH_CXX=$(which g++)
export MPICH_CC=$(which gcc)
# uv cache clean pymetis # if pymetis is not built with CC/CXX gcc/g++
uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda13 --python $(which python) --refresh --active

cd ../gt4py
uv pip install -e .
cd ..
cd dace
uv pip install -e .
cd ..
cd icon4py

# optionally, to reduce the import times, precompile all
# python modules to bytecode before creating the squashfs image
python -m compileall -j 8 -o 0 -o 1 -o 2 /dev/shm/$USER/.venv/lib/python3.13/site-packages

mksquashfs /dev/shm/$USER/.venv py_venv.squashfs \
    -no-recovery -noappend -Xcompression-level 3
