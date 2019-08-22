#!/usr/bin/env bash

# Script is for fast copy of results, tensors, checkpoint text files
# from lab servers to local machine. It reads experiments directories
# names from standard input and puts them into file 'dir_for_copy.txt'.
# File 'dirs_for_copy.txt' is copied to server where it is used for forming
# a list of files 'files_for_copy.txt'. The list for of files is used for
# creation of tar.gz archive 'for_copy.tar.gz' which is then passed to
# local machine and unpacked.
# Args:
#     1. Path to directory relative to which archive is formed. Paths to experiment
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
    printf "${line}" >> dirs_for_copy.txt
  done

  remote_expres=$(dgx1 "echo \"\${EXPRES}\"")

  scp -q dirs_for_copy.txt ${LSERV}:"${remote_expres}/$1/"

  dgx1 "bash \"\${SCRIPTS}\"create_tar_for_copy.sh \"\${EXPRES}\"/dirs_for_copy.txt"
  scp -q ${LSERV}:"${remote_expres}/$1/for_copy.tar.gz" for_copy.tar.gz
  tar -xzf for_copy.tar.gz

  cd "${starting_dir}"
}

main $1
unset main
