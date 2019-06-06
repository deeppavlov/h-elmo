#!/usr/bin/env bash

source ~/.bashrc

dpenv

setdev 0

tt=("$HOME/h-elmo/tests/experiments/correlation/nocorrloss/batch_mean/100_100short.json" \
    "$HOME/h-elmo/tests/experiments/correlation/nocorrloss/overfitting/validate_on_train.json" \
    "$HOME/h-elmo/tests/experiments/residual/nocorrloss/batch_mean/100_100short.json" \
    "$HOME/h-elmo/tests/experiments/residual/nocorrloss/overfitting/validate_on_train.json" \
    "$HOME/h-elmo/tests/experiments/resrnn/no_matrix_dim_adjustment.json" \
    "$HOME/h-elmo/tests/experiments/residual/straight1.json")

hs=("$HOME/h-elmo/tests/experiments/resrnn/small/char/hp/200.json" \
    "$HOME/h-elmo/tests/experiments/resrnn/residual/200.json")

for config in ${tt[@]}; do
  echo "Processing" $config
  python3 $TT $config --test --no_logging
done

for config in ${tt[@]}; do
  echo "Processing" $config
  python3 $HS $config --test --no_logging
done
