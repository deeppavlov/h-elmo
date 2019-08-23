#!/usr/bin/env bash


# for plotting loss and correlation in 100 100_100 500_500
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/correlation_and_loss_plots.sh "[100]@[100, 100]@[500, 500]" 100@100_100@500_500 plots
unset sorting_key


# for plotting wide correlation
cd ~/h-elmo/expres/correlation/nocorrloss
source ${PLOT}/correlation_and_loss_plots.sh "dropout 0@dropout 0.2@dropout 0.4@dropout 0.7" \
  wide/0@wide/0.2@wide/0.4@wide/0.7 wide/plots/20


# for plotting adam 100 100_100 500_500
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/correlation_and_loss_plots.sh "[100]@[100, 100]@[500, 500]" \
  adam/100@batch_mean/100_100short@wide/0 adam/plots
unset sorting_key


# for plotting loss - correlation 100 100_100 500_500
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/loss_corr_plot.sh "[100]@[100, 100]@[500, 500]" 100@100_100@500_500 plots
unset sorting_key


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers text8
cd ~/h-elmo/expres/correlation/nocorrloss/text8
xselect_params=( "-S" "1.5" "3.5" )
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'adam' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  adam/100@adam/100_100@adam/500_500@sgd/100@sgd/100_100@sgd/500_500 plots
unset sorting_key
unset xselect_params


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers text8
# with shuffled axvspan
cd ~/h-elmo/expres/correlation/nocorrloss/text8
xselect_params=( "-S" "1.5" "3.5" )
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'adam' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
additional_artists_str=( "-a"\
  "../shuffled/text8/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  adam/100@adam/100_100@adam/500_500@sgd/100@sgd/100_100@sgd/500_500 plots
unset sorting_key
unset xselect_params
unset additional_artists_str


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers enwiki1G
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'adam' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  adam/100@batch_mean/100_100short@wide/0@sgd/100@sgd/100_100@sgd/500_500 enwiki1G/plots
unset sorting_key


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers enwiki1G
# with shuffled axvspan
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'adam' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
additional_artists_str=( "-a"\
  "shuffled/enwiki1G/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  adam/100@batch_mean/100_100short@wide/0@sgd/100@sgd/100_100@sgd/500_500 enwiki1G/plots
unset sorting_key
unset additional_artists_str


# for plotting loss - corrrelation 100 100_100 500_500 for adam enwiki1G-text8
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'enwiki1G' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] enwiki1G@[100, 100] enwiki1G@[500, 500] enwiki1G@[100] text8@[100, 100] text8@[500, 500] text8" \
  adam/100@batch_mean/100_100short@wide/0@text8/adam/100@text8/adam/100_100@text8/adam/500_500 \
  enwiki1G-text8/plots/adam
unset sorting_key


# for sgd
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'enwiki1G' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] enwiki1G@[100, 100] enwiki1G@[500, 500] enwiki1G@[100] text8@[100, 100] text8@[500, 500] text8" \
  sgd/100@sgd/100_100@sgd/500_500@text8/sgd/100@text8/sgd/100_100@text8/sgd/500_500 enwiki1G-text8/plots/sgd
unset sorting_key


# for plotting loss - corrrelation 100 100_100 500_500 for adam enwiki1G-text8
# with shuffled vspans
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'enwiki1G' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
additional_artists_str=( "-a"\
  "shuffled/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] enwiki1G@[100, 100] enwiki1G@[500, 500] enwiki1G@[100] text8@[100, 100] text8@[500, 500] text8" \
  adam/100@batch_mean/100_100short@wide/0@text8/adam/100@text8/adam/100_100@text8/adam/500_500 \
  enwiki1G-text8/plots/adam
unset sorting_key
unset additional_artists_str


# for sgd
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'enwiki1G' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
additional_artists_str=( "-a"\
  "shuffled/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "[100] enwiki1G@[100, 100] enwiki1G@[500, 500] enwiki1G@[100] text8@[100, 100] text8@[500, 500] text8" \
  sgd/100@sgd/100_100@sgd/500_500@text8/sgd/100@text8/sgd/100_100@text8/sgd/500_500 enwiki1G-text8/plots/sgd
unset sorting_key
unset additional_artists_str


# for sgd vary learning rate loss - corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_lr/text8/sgd/100
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "learning rate 3@learning rate 1@learning rate 0.3@learning rate 0.1@learning rate 0.03@learning rate 0.01" \
  3@1@0.3@0.1@0.03@0.01 plots
unset sorting_key
unset additional_artists_str
unset lw_params


# for sgd vary learning rate loss AND corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_lr/text8/sgd/100
sorting_key="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "learning rate 3@learning rate 1@learning rate 0.3@learning rate 0.1@learning rate 0.03@learning rate 0.01" \
  3@1@0.3@0.1@0.03@0.01 plots
unset sorting_key


# for sgd vary batch size loss - corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_bs/text8/sgd/100
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "batch size 1024@batch size 512@batch size 256@batch size 128@batch size 64@batch size 32@batch size 20@batch size 10" \
  1024@512@256@128@64@32@20@10 plots
unset sorting_key
unset additional_artists_str
unset lw_params


# for sgd vary batch size loss AND corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_bs/text8/sgd/100
sorting_key="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "batch size 1024@batch size 512@batch size 256@batch size 128@batch size 64@batch size 32@batch size 20@batch size 10" \
  1024@512@256@128@64@32@20@10 plots
unset sorting_key


# for sgd vary number of unrollings loss - corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_unr/text8/sgd/100
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "sequence length 1000@sequence length 400@sequence length 200@sequence length 100@sequence length 50@sequence length 20@sequence length 10@sequence length 5" \
  1000@400@200@100@50@20@10@5 plots
unset sorting_key
unset additional_artists_str
unset lw_params


# for sgd vary number of unrollings loss AND corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_unr/text8/sgd/100
sorting_key="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "sequence length 1000@sequence length 400@sequence length 200@sequence length 100@sequence length 50@sequence length 20@sequence length 10@sequence length 5" \
  1000@400@200@100@50@20@10@5 plots
unset sorting_key

# batch copy
cd ${EXPRES}/correlation/batch
function gen_exp_dirs () {
  local dt
  local opt
  local net

  local exp

  local dp
  local bs
  local unr

  for shuffled_str in shuffled/ ""
  do
    for dt in enwiki1G text8
    do
      for opt in adam sgd
      do
        for net in 100 100_100 500_500
        do
          echo "${shuffled_str}${dt}/${opt}/${net}"
        done
      done
    done
  done

#  for dp in 0.4 0.7 0
#  do
#    echo "long_dropout/${dp}"
#  done

  for exp in validate_on_train validate_on_train_swap
  do
    echo "overfitting/${exp}"
  done

  for dp in dp0.7 dp0
  do
    echo "second_layer/${dp}"
  done

  for bs_dir in vary_bs # vary_bs_long
  do
    for bs in 10 20 32 64 128 256 512 1024
    do
      echo "${bs_dir}/text8/sgd/100/${bs}"
    done
  done

  for dp in 0 0.2 0.3 0.4 0.5 0.6 0.7
  do
    echo "vary_dropout/${dp}"
  done

  for lr in 0.01 0.001 0.0001 0.00001 0.002 0.0003 0.00003 0.005
  do
    echo "vary_lr/text8/adam/100/${lr}"
  done

  for lr in 0.1 0.01 0.3 0.03 1 3
  do
    echo "vary_lr/text8/sgd/100/${lr}"
  done

  for dp in 0 0.2 0.4 0.7
  do
    echo "wide/vary_dropout/${dp}"
  done

#  for unr in 5 10 20 50 100 200 400 1000
#  do
#    echo "vary_unr/text8/sgd/100/${unr}"
#  done
}
source ${SCRIPTS}/scp_results_and_tensors_fast_2.sh < <(gen_exp_dirs)
unset gen_exp_dirs
