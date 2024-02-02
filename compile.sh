#!/bin/bash
set -e -x

export LIB=.

py2fgen icon4pytools.py2f.wrappers.square_wrapper square
gfortran -c square_wrapper_plugin.f90 .
gfortran -I$LIB -Wl,-rpath=$LIB -L$LIB  square_wrapper_plugin.f90 run_square.f90 -lsquare_wrapper_plugin
