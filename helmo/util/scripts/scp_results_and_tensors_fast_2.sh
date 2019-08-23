#!/usr/bin/env bash

# Script is for fast copy of results, tensors, checkpoint text files
# from lab servers to local machine. It reads experiments directories
# names from standard input and puts them into file 'dirs_for_copy.txt'.
# File 'dirs_for_copy.txt' is copied to server where it is used for forming
# a list of files 'files_for_copy.txt'. The list for of files is used for
# creation of tar.gz archive 'for_copy.tar.gz' which is then passed to
# local machine and unpacked.
# Args:
#     1. Path to directory relative to which archive is formed. Paths to experiments
#        have to be given relative to this directory. Path to this directory is
#        given relative to expres
# Paths to experiments are give relative to directory provided as first argument to this script


function main() {
  local starting_dir=$(pwd)

  mkdir -p ${EXPRES}/$1
  cd ${EXPRES}/$1

  > dirs_for_copy.txt

  local line
  while read line
  do
    printf "${line}\n" >> dirs_for_copy.txt
  done

  local remote_expres_command=$(dgx1 "cat ~/.bashrc | grep EXPRES")
  local remote_expres_descr=${remote_expres_command#"export EXPRES="}
  local remote_expres=$(dgx1 "echo ${remote_expres_descr}")

  local remote_scripts_command=$(dgx1 "cat ~/.bashrc | grep SCRIPTS")
  local remote_scripts_descr=${remote_scripts_command#"export SCRIPTS="}
  local remote_scripts=$(dgx1 "echo ${remote_scripts_descr}")

  scp -q dirs_for_copy.txt ${LSERV}:"${remote_expres}/$1/"

  dgx1 "source ${remote_scripts}/create_tar_for_copy.sh ${remote_expres}/$1/dirs_for_copy.txt ${remote_expres}/$1"
  scp -q ${LSERV}:"${remote_expres}/$1/for_copy.tar.gz" for_copy.tar.gz
  tar -xzf for_copy.tar.gz

  cd "${starting_dir}"
}

main $1
unset main
