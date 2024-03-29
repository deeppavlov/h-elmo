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

  local remote_scripts_command=$(cat ~/.bashrc | grep SCRIPTS)
  local remote_scripts_descr=${remote_scripts_command#"export SCRIPTS="}
  local remote_scripts=$(echo ${remote_scripts_descr})

  local starting_dir="$(pwd)"
  local line
  local input_file

  input_file="$(realpath $1)"

  cd "$2"

  while read line
  do
    bash "${remote_scripts}/form_list_of_files_for_copy_1_dir.sh" "${line}" \
        files_for_copy.txt
  done < "${input_file}"

  local -a file_names
  mapfile -t file_names \
      < <(while read line; do echo "${line}"; done < files_for_copy.txt)

  tar -czf for_copy.tar.gz "${file_names[@]}"

  rm files_for_copy.txt

  cd "${starting_dir}"
}

main "$1" "$2"
unset main
