#!/usr/local/bin/bash -l

run_twice() {
  cmd=$@
  $cmd &
  $cmd &
  wait
}

# Use the function
run_twice "$@"
