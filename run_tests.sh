#!/usr/bin/env bash

source ~/.bashrc

dpenv

setdev 0

python3 $TT ~/h-elmo/tests/experiments/correlation/nocorrloss/batch_mean/100_100short.json --test --no_logging
python3 $TT ~/h-elmo/tests/experiments/correlation/nocorrloss/overfitting/validate_on_train.json --test --no_logging
python3 $TT ~/h-elmo/tests/experiments/residual/nocorrloss/batch_mean/100_100short.json --test --no_logging
python3 $TT ~/h-elmo/tests/experiments/residual/nocorrloss/overfitting/validate_on_train.json --test --no_logging
python3 $TT ~/h-elmo/tests/experiments/resrnn/no_matrix_dim_adjustment.json --test --no_logging
python3 $TT ~/h-elmo/tests/experiments/residual/straight1.json --test --no_logging

python3 $HS ~/h-elmo/tests/experiments/resrnn/small/char/hp/200.json --test --no_logging
python3 $HS ~/h-elmo/tests/experiments/resrnn/residual/200.json --test --no_logging
