#!/usr/bin/env bash

# This script is for for creating file containing
# paths to files with results and tensors. If
# output file already exists paths are appended to
# file.
# Args:
#     1. Output file name
#
# Directories for parsing are passed in stdin

function main () {
  while read line
  do
    bash ${SCRIPTS}/form_list_of_files_for_copy_1_dir.sh "${line}" "$1"
  done
}

main "$1"
unset main
