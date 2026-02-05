#!/bin/bash

set -e

# Set necessasry flags for compilation
source setup_env.sh

date

unset PYTHONPATH

# Install uv locally
export PATH="$PWD/bin:$PATH"
if [ ! -x "$PWD/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$PWD/bin" sh
else
    echo "# uv already installed at $PWD/bin/uv"
fi

# Install icon4py, gt4py, DaCe and other basic dependencies using uv
uv sync --extra all --python $(which python3.12)

# Activate virtual environment
source .venv/bin/activate

# Compatibility for Beverin
mpi4py_ver=$(uv pip show mpi4py | awk '/Version:/ {print $2}')
uv pip uninstall mpi4py && uv pip install --no-binary mpi4py "mpi4py==$mpi4py_ver"
pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.0.2/simple

# Install the requirements for rocprofiler-compute so we can run the profiler from the same environment
uv pip install -r /user-environment/linux-zen3/rocprofiler-compute-7.1.0-rjjjgkz67w66bp46jw7bvlfyduzr6vhv/libexec/rocprofiler-compute/requirements.txt

################################################################################
# Serialbox / libstdc++ auto-discovery
################################################################################

# 1) Fix for the current script run: find libSerialboxC.so, ask ldd which
#    libstdc++.so it uses, and prepend that directory to LD_LIBRARY_PATH.
serialbox_so="$(find "$PWD/.venv" -maxdepth 7 -type f -name 'libSerialboxC.so*' 2>/dev/null | head -n1 || true)"
if [ -n "$serialbox_so" ] && [ -f "$serialbox_so" ]; then
    libstdcpp_path="$(ldd "$serialbox_so" 2>/dev/null | awk '/libstdc\+\+\.so/ {print $3; exit}')"
    if [ -n "$libstdcpp_path" ] && [ -f "$libstdcpp_path" ]; then
        libstdcpp_dir="$(dirname "$libstdcpp_path")"

        case ":$LD_LIBRARY_PATH:" in
            *":$libstdcpp_dir:"*) : ;;
            *) export LD_LIBRARY_PATH="${libstdcpp_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac

        echo "# Serialbox library      : $serialbox_so"
        echo "# Serialbox libstdc++    : $libstdcpp_path"
        echo "# Adding libstdc++ dir to LD_LIBRARY_PATH: $libstdcpp_dir"
    else
        echo "# WARNING: Could not determine libstdc++.so used by $serialbox_so"
    fi
else
    echo "# NOTE: libSerialboxC.so not found under .venv yet (is serialbox installed?)."
fi

# 2) Persist this logic into .venv/bin/activate so every future activation
#    automatically discovers Serialbox and its libstdc++ and updates LD_LIBRARY_PATH.
cat >> .venv/bin/activate <<'EOF'

# Added automatically so Serialbox can always find the right libstdc++
if [ -n "$VIRTUAL_ENV" ]; then
    serialbox_so=$(find "$VIRTUAL_ENV" -maxdepth 7 -type f -name 'libSerialboxC.so*' 2>/dev/null | head -n 1)
    if [ -n "$serialbox_so" ] && [ -f "$serialbox_so" ]; then
        libstdcpp_path=$(ldd "$serialbox_so" 2>/dev/null | awk '/libstdc\+\+\.so/ {print $3; exit}')
        if [ -n "$libstdcpp_path" ] && [ -f "$libstdcpp_path" ]; then
            libstdcpp_dir=$(dirname "$libstdcpp_path")
            case ":$LD_LIBRARY_PATH:" in
                *:"$libstdcpp_dir":*) : ;;
                *) export LD_LIBRARY_PATH="$libstdcpp_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
            esac
        fi
    fi
fi

EOF

################################################################################

echo "# install done"
date
