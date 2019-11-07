#!/usr/bin/env bash

# This script is for plotting loss - correlation dependency
# Args:
#   1. string of artist labels separated by at signs
#   2. string of experiment result dirs separated by at signs
#   3. path to dir where results will be stored
#   sorting_key: definition of a function for sorting lines in plot_data

function main () {

  if test -z "${sorting_key}"
  then
    local sorting_str=""
    local exec_scr_str=""
  else
    local sorting_str="-k=${sorting_key}"
    local exec_scr_str="-e=$3/loss-corr_data_exec.py"
  fi

  oldIFS="$IFS"
  IFS="@"
  local labels=($1)
  local exp_dirs=($2)

  IFS="${oldIFS}"

  mkdir -p $3
  local -a x_sources
  local -a y_sources
  local -a yerr_sources
  local exp_dir
  for exp_dir in "${exp_dirs[@]}"
  do
    x_sources+=("${exp_dir}/mean/loss.txt")
    y_sources+=("${exp_dir}/mean/corr/mean.pickle")
    yerr_sources+=("${exp_dir}/mean/corr/stddev.pickle")
  done
  python3 ${PLOT}/merge_data_from_diff_sources.py --xsrc "${x_sources[@]}" \
    --ysrc "${y_sources[@]}" --yerrsrc "${yerr_sources[@]}" --colx 1 \
    --xerrcol 2 --labels "${labels[@]}" -o $3/loss-corr_data.pickle \
    --no_sort "${sorting_str}"
  python3 ${PLOT}/plot_from_pickle.py $3/loss-corr_data.pickle -x loss \
    -y "mean square correlation" \
    -X linear -o $3/loss_corr_plot -t bar -d best -O -s png -r 900 -g -w both \
    --no_line ${exec_scr_str} \
    ${additional_artists_str:+"${additional_artists_str[@]}"} \
    ${xselect_params:+"${xselect_params[@]}"} ${lw_params:+"${lw_params[@]}"}
}


main "$1" "$2" "$3"
unset main
