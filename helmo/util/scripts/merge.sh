#!/usr/bin/env bash

current_dir=$(pwd)

cd $1

for dropout_dir in 100_100short*; do
  cd "$dropout_dir"
  echo "Processing $dropout_dir"
  python3 ~/h-elmo/helmo/util/scripts/merge.py
  cd ..
done

cd ${current_dir}
