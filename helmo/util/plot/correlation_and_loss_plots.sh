#!/usr/bin/env bash

# This script is for drawing a plot with steps on
# horizontal axis and correlation on vertical axis.
# Plot data is created and put in the same directory with
# plot.

function main () {
  oldIFS="$IFS"
  IFS="@"
  local labels=($1)
  local exp_dirs=($2)
  local loss_file="${exp_dirs[0]}"/0/results/loss_valid.txt
  local -a mean_corr
  local -a err_corr
  local -a loss_files
  local exp_dir
  for exp_dir in "${exp_dirs[@]}"
  do
    mean_corr+=(${exp_dir}/mean/corr/mean.pickle)
    err_corr+=(${exp_dir}/mean/corr/stddev.pickle)
    loss_files+=(${exp_dir}/mean/loss.txt)
  done
  python3 ${SCRIPTS}/plot_data_from_pickle.py -l ${labels[*]} -s ${loss_file} \
    -m ${mean_corr[*]} -d ${err_corr[*]} -n -o $3/corr_plot_data.pickle
  python3 ${SCRIPTS}/plot_from_pickle.py $3/corr_plot_data.pickle -x step -y correlation \
    -X log -o $3/corr_plot -t fill -d best -O -s png -r 900 -g -w both
  python3 ${SCRIPTS}/plot_data_from_txt.py ${loss_files[*]} -l ${labels[*]} -x 0 -y 1 -e 2 \
    -o $3/loss_plot_data.pickle
  python3 ${SCRIPTS}/plot_from_pickle.py $3/loss_plot_data.pickle -x step -y correlation \
    -X log -o $3/loss_plot -t fill -d best -O -s png -r 900 -g -w both
}


main $1 $2 $3
