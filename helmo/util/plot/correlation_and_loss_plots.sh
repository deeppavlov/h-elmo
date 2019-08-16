#!/usr/bin/env bash

# This script is for drawing a plot with steps on
# horizontal axis and correlation on vertical axis.
# Plot data is created and put in the same directory with
# plot.

# Args:
#   labels: string of artist labels separated by at signs
#   exp_dirs: string of experiment result dirs separated by at signs
#   Path to dir where results will be stored

function main () {
  oldIFS="$IFS"
  IFS="@"
  local labels=($1)
  local exp_dirs=($2)
  IFS=${oldIFS}
  mkdir -p $3
  local -a mean_corr
  local -a err_corr
  local -a step_corr
  local -a loss_files
  local exp_dir
  for exp_dir in "${exp_dirs[@]}"
  do
    mean_corr+=(${exp_dir}/mean/corr/mean.pickle)
    err_corr+=(${exp_dir}/mean/corr/stddev.pickle)
    step_corr+=(${exp_dir}/0/results/loss_valid.txt)
    loss_files+=(${exp_dir}/mean/loss.txt)
  done

  python3 ${PLOT}/plot_data_from_pickle.py -l ${labels[*]} -s ${step_corr[*]} \
    -m ${mean_corr[*]} -d ${err_corr[*]} -n -o $3/corr_plot_data.pickle
  python3 ${PLOT}/plot_from_pickle.py $3/corr_plot_data.pickle -x step -y correlation \
    -X log -o $3/corr_plot -t fill -d best -O -s png -r 900 -g -w both
  python3 ${PLOT}/plot_data_from_txt.py ${loss_files[*]} -l ${labels[*]} -x 0 -y 1 -e 2 \
    -o $3/loss_plot_data.pickle
  python3 ${PLOT}/plot_from_pickle.py $3/loss_plot_data.pickle -x step -y loss \
    -X log -o $3/loss_plot -t fill -d best -O -s png -r 900 -g -w both
}


main $1 $2 $3
