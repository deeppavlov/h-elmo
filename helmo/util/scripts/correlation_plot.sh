#!/usr/bin/env bash

# This script is for drawing a plot with steps on
# horizontal axis and correlation on vertical axis.
# Plot data is created and put in the same directory with
# plot.

function main () {
  local labels=$(echo $1 | tr @ " ")
  local exp_dirs=$(echo $2 | tr @ " ")
  python3 ${SCRIPTS}/plot_data_from_pickle.py
}