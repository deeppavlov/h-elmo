#!/bin/bash

# applied to results of dropout experiments

current_dir=$(pwd)

cd $1

for dropout_dir in 100_100short*; do
  cd "$dropout_dir"
  echo "Processing $dropout_dir"
  pickle_files=()
  for launch in $(ls | grep -E '^[0-9]+$'); do
    pickle_files+=(${launch}/tensors/valid_pickle_mean_tensors/correlation_valid.pickle)
  done
  python3 ~/h-elmo/helmo/util/scripts/average_pickle_values.py ${pickle_files[@]} --preprocess $2
  cd ..
done

cd ${current_dir}
