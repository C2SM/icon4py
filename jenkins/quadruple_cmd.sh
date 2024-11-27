#!/usr/local/bin/bash -l

quadruple_command() {
  cmd=$@
  $cmd &
  pid1=$!
  $cmd &
  pid2=$!
  $cmd &
  pid3=$!
  $cmd &
  pid4=$!

  wait $pid1
  status1=$?
  wait $pid2
  status2=$?
  wait $pid3
  status3=$?
  wait $pid4
  status4=$?

  if [ $status1 -ne 0 ] || [ $status2 -ne 0 ] || [ $status3 -ne 0 ] || [ $status4 -ne 0 ]; then
    exit 1
  fi
}

quadruple_command "$@"
