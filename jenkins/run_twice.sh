#!/usr/local/bin/bash -l

# Function to run a command twice concurrently
run_twice() {
  cmd=$@

  # Run the command in the background
  $cmd &

  # Run the command again in the background
  $cmd &

  # Wait for all background jobs to finish
  wait
}

# Use the function
run_twice "$@"
