#!/usr/bin/env bash

# This script is for for creating file containing
# paths to files with results and tensors. If
# output file already exists paths are appended to
# file.
# Args:
#     1. Path to dir with experiments
#     2. Output file name

function main () {
  local -a launch_dirs
  local -a txt_files
  mapfile -t launch_dirs < <(ls $1 | grep -E '^(0|[1-9][0-9]*)$')
  mapfile -t txt_files < <(ls $1 | grep -E '\.txt$')
  echo "$2"
  local txt_f
  for txt_f in "${txt_file[@]}"
  do
    printf "$1/${txt_f}\n" >> "$2"
  done

  local -a checkpoint_txt_files
  local launch_dir
  local ckpt_str
  for launch_dir in "${launch_dirs[@]}"
  do
    printf "$1/${launch_dir}/results\n" >> "$2"
    printf "$1/${launch_dir}/tensors\n" >> "$2"
    [ -d "$1/${launch_dir}/checkpoints/all_vars" ] && ckpt_path=checkpoints/all_vars || ckpt_path=checkpoints
    mapfile -t checkpoint_txt_files < <(ls $1/${launch_dir}/${ckpt_path} | grep -E '\.txt$')
    for ckpt_txt_file in "${checkpoint_txt_files[@]}"
    do
      printf "$1/${launch_dir}/${ckpt_path}/${ckpt_txt_file}\n" >> "$2"
    done
  done
}

main "$1" "$2"
unset main
