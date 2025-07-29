#!/bin/bash
# Example mps-wrapper.sh usage:
# > srun --cpu-bind=socket [srun args] mps-wrapper.sh [cmd] [cmd args]

# only this path is supported by MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un)
# Launch MPS from a single rank per node
if [[ $SLURM_LOCALID -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
fi

# set cuda device
# TODO(iomaganaris): this needs to be adapted when we have node sharing and for `pytest-xdist`
export CUDA_VISIBLE_DEVICES=$(($SLURM_LOCALID / 4))

# Wait for MPS to start
sleep 1
# Run the command
"$@"
result=$?
# Quit MPS control daemon before exiting
if [[ $SLURM_LOCALID -eq 0 ]]; then
    echo quit | nvidia-cuda-mps-control
fi
exit $result
