#!/bin/bash
set -e -x

export LIB=.

py2fgen icon4py.model.atmosphere.dycore.compute_airmass compute_airmass
gfortran -c compute_airmass_plugin.f90 .
gfortran -I$LIB -Wl,-rpath=$LIB -L$LIB  compute_airmass_plugin.f90 run_compute_airmass.f90 -lcompute_airmass_plugin
