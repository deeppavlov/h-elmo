#!/usr/bin/env bash

# This script is for copying collected tensors, results and
# testing results from servers

# alias dgx1="ssh peganov@dgx1.ipavlov.mipt.ru"



function main () {
  mkdir -p $1
  local current_dir=$(pwd)
  cd $1

  local abs_path=$(pwd)
  local relative_to_expres=${abs_path#"$EXPRES/"}
  local rem_expres_command=$(dgx1 "cat ~/.bashrc | grep EXPRES")
  local rem_expres_descr=${rem_expres_command#"export EXPRES="}
  local rem_expres=$(dgx1 "echo ${rem_expres_descr}")
  local remote_path=${rem_expres}/${relative_to_expres}
  local launch_dirs=$(dgx1 "ls ${remote_path} | grep -E '^(0|[1-9][0-9]*)$'")
  local txt_files=$(dgx1 "ls ${remote_path} | grep -E '\.txt$'")

  for f in ${txt_files}
  do
    scp ${LSERV}:${remote_path}/${f} ./
  done

  for d in ${launch_dirs}
  do
    mkdir -p ${d}/checkpoints/all_vars
    scp -r  ${LSERV}:${remote_path}/${d}/tensors ${d}/
    scp -r  ${LSERV}:${remote_path}/${d}/results ${d}/
    local checkpoint_txt=$(dgx1 "ls ${remote_path}/${d}/checkpoints/all_vars | grep -E '\.txt$'")
    for ckpt_txt in ${checkpoint_txt}
    do
      scp -r  ${LSERV}:${remote_path}/${d}/checkpoints/all_vars/${ckpt_txt} ${d}/checkpoints/all_vars
    done
  done

  cd ${current_dir}
}

main $1
unset main
