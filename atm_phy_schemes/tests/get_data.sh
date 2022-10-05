#!/bin/bash

#Download serialized data from IAC FTP
wget -r --no-parent -nH -nc --cut-dirs=3 -q --show-progress ftp://iacftp.ethz.ch/pub_read/davidle/ser_data_icon_graupel/ser_data/
