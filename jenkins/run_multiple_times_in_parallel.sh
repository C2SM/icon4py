#!/usr/local/bin/bash -l

quadruple_command() {
  cmd=$@
  $cmd &
  $cmd &
  $cmd &
  $cmd &
  wait
}

quadruple_command "$@"
