#!/usr/bin/env bash

# This script is for copying collected tensors, results and
# testing results from servers


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

  echo ${remote_path}

  local -a archived

  local f
  for f in ${txt_files}
  do
    archived+=("\"${f}\"")
  done

  local d
  local ckpt_txt
  for d in ${launch_dirs}
  do
    dgx1 "[ -d \"${remote_path}/${d}/checkpoints/all_vars\" ]" && ckpt_str=checkpoints/all_vars || ckpt_str=checkpoints
    archived+=("\"${d}/tensors\"")
    archived+=("\"${d}/results\"")
    local checkpoint_txt=$(dgx1 "ls ${remote_path}/${d}/${ckpt_str} | grep -E '\.txt$'")
    for ckpt_txt in ${checkpoint_txt}
    do
      archived+=("\"${d}/${ckpt_str}/${ckpt_txt}\"")
    done
  done
  dgx1 "cd \"${remote_path}\";tar -czf \"tmp.tar.gz\" ${archived[@]}"
  scp -q ${LSERV}:${remote_path}/tmp.tar.gz ./

  cd ${current_dir}
}

main $1
unset main
