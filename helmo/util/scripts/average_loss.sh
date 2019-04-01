#!/usr/bin/env bash

current_dir=$(pwd)

cd $1

for dropout_dir in 100_100short*; do
  cd "$dropout_dir"
  echo "Processing $dropout_dir"
  loss_files=()
  for launch in $(ls | grep -E '^[0-9]+$'); do
    loss_files+=(${launch}/results/loss_valid.txt)
  done
  python3 ~/h-elmo/helmo/util/scripts/average_txt.py ${loss_files[@]} -o loss.txt
  cd ..
done

cd ${current_dir}
