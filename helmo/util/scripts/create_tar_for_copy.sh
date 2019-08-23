#!/usr/bin/env bash

# This script is for creating archive containing
# experiment results and tensors. The first argument is
# path to file with list of experiment directories. Paths
# to experiment directories are given relative to path
# provided in second argument.
# Args:
#     1. Path to file with experiment directories
#     2. Path to directory where archive should be stored


function main () {
  local starting_dir="$(pwd)"
  local line
  local input_file

  input_file="$(realpath $1)"

  cd "$2"

  echo $(pwd)
  echo ${input_file}

  while read line
  do
    echo ${line}
    bash ${SCRIPTS}/form_list_of_files_for_copy_1_dir.sh "${line}" files_for_copy.txt
  done < "${input_file}"

  ls

  local -a file_names
  mapfile -t file_names < <(while read line; do echo "${line}"; done < files_for_copy.txt)

  tar -czf for_copy.tar.gz "${file_names[@]}"

  cd "${starting_dir}"
}

main "$1" "$2"
unset main
