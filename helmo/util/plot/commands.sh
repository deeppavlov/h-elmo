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


# for sgd vary learning rate loss - corr plots ADAM
cd ~/h-elmo/expres/correlation/nocorrloss/vary_lr/text8/adam/100
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "learning rate 0.01@learning rate 0.005@learning rate 0.002@learning rate 0.001@learning rate 0.0003@learning rate 0.0001@learning rate 0.00003@learning rate 0.00001" \
  0.01@0.005@0.002@0.001@0.0003@0.0001@0.00003@0.00001 plots
unset sorting_key
unset additional_artists_str
unset lw_params


# for sgd vary learning rate loss AND corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_lr/text8/adam/100
sorting_key="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "learning rate 0.01@learning rate 0.005@learning rate 0.002@learning rate 0.001@learning rate 0.0003@learning rate 0.0001@learning rate 0.00003@learning rate 0.00001" \
  0.01@0.005@0.002@0.001@0.0003@0.0001@0.00003@0.00001 plots
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

  for unr in 5 10 20 50 100 200 400 1000
  do
    echo "vary_unr/text8/sgd/100/${unr}"
  done
}
source ${SCRIPTS}/scp_results_and_tensors_fast_2.sh correlation/batch < <(gen_exp_dirs)
unset gen_exp_dirs

# process experiment results
cd ${EXPRES}
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

  for unr in 5 10 20 50 100 200 400 1000
  do
    echo "vary_unr/text8/sgd/100/${unr}"
  done
}
while read line
do
  source ${SCRIPTS}/process_corr_exp_results.sh "correlation/batch/${line}"
done < <(gen_exp_dirs)
unset gen_exp_dirs


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers text8 BATCH
cd ~/h-elmo/expres/correlation/batch/text8
additional_artists_str=( "-a"\
  "../../nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
# xselect_params=( "-S" "1.5" "3.5" )
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
unset additional_artists_str


# enwiki1G BATCH
cd ~/h-elmo/expres/correlation/batch/enwiki1G
additional_artists_str=( "-a"\
  "../../nocorrloss/shuffled/enwiki1G/noise_best_loss_axvspan.pickle" )
# xselect_params=( "-S" "1.5" "3.5" )
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
unset additional_artists_str


# for plotting loss - corrrelation 100 100_100 500_500 for adam enwiki1G-text8 BATCH
cd ~/h-elmo/expres/correlation/batch
additional_artists_str=( "-a"\
  "../nocorrloss/shuffled/noise_best_loss_axvspan.pickle" )
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
  enwiki1G/adam/100@enwiki1G/adam/100_100@enwiki1G/adam/500_500@text8/adam/100@text8/adam/100_100@text8/adam/500_500 \
  enwiki1G-text8/plots/adam
unset sorting_key
unset additional_artists_str


# for sgd BATCH
cd ~/h-elmo/expres/correlation/batch
additional_artists_str=( "-a"\
  "../nocorrloss/shuffled/noise_best_loss_axvspan.pickle" )
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
  enwiki1G/sgd/100@enwiki1G/sgd/100_100@enwiki1G/sgd/500_500@text8/sgd/100@text8/sgd/100_100@text8/sgd/500_500 \
  enwiki1G-text8/plots/sgd
unset sorting_key
unset additional_artists_str


# Overfitting
cd ~/h-elmo/expres/correlation/batch/overfitting
cd validate_on_train
python3 "${PLOT}/plot_data_from_pickle.py" -l validation train \
  -s 0/results/loss_valid_train.txt 0/results/loss_valid.txt -m 0/tensors/valid/pickle_mean_tensors/correlation.pickle \
  0/tensors/valid_train/pickle_mean_tensors/correlation.pickle -n -p sqrt -o plots/corr_plot_data.pickle
python3 "${PLOT}/plot_data_from_txt.py" 0/results/loss_valid.txt 0/results/loss_valid_train.txt \
  -l validation train -o plots/loss_plot_data.pickle -n
python3 "${PLOT}/plot_from_pickle.py" plots/corr_plot_data.pickle -y "mean square correlation" -X symlog -t noerr \
  --lgd best -s png -r 900 -g -w both -o plots/correlation_plot
python3 "${PLOT}/plot_from_pickle.py" plots/loss_plot_data.pickle -y loss -X symlog -t noerr \
  --lgd best -s png -r 900 -g -w both -o plots/loss_plot
cd ../validate_on_train_swap
python3 "${PLOT}/plot_data_from_pickle.py" -l validation train \
  -s 0/results/loss_valid_train.txt 0/results/loss_valid.txt -m 0/tensors/valid/pickle_mean_tensors/correlation.pickle \
  0/tensors/valid_train/pickle_mean_tensors/correlation.pickle -n -p sqrt -o plots/corr_plot_data.pickle
python3 "${PLOT}/plot_data_from_txt.py" 0/results/loss_valid.txt 0/results/loss_valid_train.txt \
  -l validation train -o plots/loss_plot_data.pickle -n
python3 "${PLOT}/plot_from_pickle.py" plots/corr_plot_data.pickle -y "mean square correlation" -X symlog -t noerr \
  --lgd best -s png -r 900 -g -w both -o plots/correlation_plot
python3 "${PLOT}/plot_from_pickle.py" plots/loss_plot_data.pickle -y loss -X symlog -t noerr \
  --lgd best -s png -r 900 -g -w both -o plots/loss_plot


# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers text8 BATCH ADAM_INIT3
cd ~/h-elmo/expres/correlation/batch/text8
additional_artists_str=( "-a"\
  "../../nocorrloss/shuffled/text8/noise_best_loss_axvspan.pickle" )
# xselect_params=( "-S" "1.5" "3.5" )
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
  adam_init3/100@adam_init3/100_100@adam_init3/500_500@sgd/100@sgd/100_100@sgd/500_500 plots_adam_init3
unset sorting_key
unset xselect_params
unset additional_artists_str


# enwiki1G BATCH ADAM_INIT3
cd ~/h-elmo/expres/correlation/batch/enwiki1G
additional_artists_str=( "-a"\
  "../../nocorrloss/shuffled/enwiki1G/noise_best_loss_axvspan.pickle" )
# xselect_params=( "-S" "1.5" "3.5" )
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
  adam_init3/100@adam_init3/100_100@adam_init3/500_500@sgd/100@sgd/100_100@sgd/500_500 plots_adam_init3
unset sorting_key
unset xselect_params
unset additional_artists_str


# for plotting loss - corrrelation 100 100_100 500_500 for adam enwiki1G-text8 BATCH ADAM_INIT3
cd ~/h-elmo/expres/correlation/batch
additional_artists_str=( "-a"\
  "../nocorrloss/shuffled/noise_best_loss_axvspan.pickle" )
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
  enwiki1G/adam_init3/100@enwiki1G/adam_init3/100_100@enwiki1G/adam_init3/500_500@text8/adam_init3/100@text8/adam_init3/100_100@text8/adam_init3/500_500 \
  enwiki1G-text8/plots_adam_init3/adam
unset sorting_key
unset additional_artists_str


# for plotting loss and RMS of hidden state elements 100, 100_100, 500_500 enwiki1G BATCH ADAM - SGD
cd ~/h-elmo/expres/hidden_state_rms
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'ADAM' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/rms_and_loss_plots.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  enwiki1G/adam/100@enwiki1G/adam/100_100@enwiki1G/adam/500_500@enwiki1G/sgd/100@enwiki1G/sgd/100_100@enwiki1G/sgd/500_500 \
  enwiki1G/plots_adam_sgd/loss_and_rms
unset sorting_key
unset additional_artists_str


# for plotting loss and RMS of hidden state elements 100, 100_100, 500_500 text8 BATCH ADAM - SGD
cd ~/h-elmo/expres/hidden_state_rms
sorting_key2="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'ADAM' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
sorting_key="${sorting_key2}" \
  source ${PLOT}/rms_and_loss_plots.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  text8/adam/100@text8/adam/100_100@text8/adam/500_500@text8/sgd/100@text8/sgd/100_100@text8/sgd/500_500 \
  text8/plots_adam_sgd/loss_and_rms
unset sorting_key
unset additional_artists_str


# Average RMS of hidden states
cd ~/h-elmo/expres/hidden_state_rms
for ds in enwiki1G text8; do
  for opt in adam sgd; do
    for nn in 100 100_100 500_500; do
      cd ${ds}/${opt}/${nn}
      mkdir rms
      python3 ${SCRIPTS}/average_pickle_values.py {0..19}/tensors/valid/pickle_mean_tensors/rms1.pickle \
          --stddev rms/stddev.pickle --mean rms/mean.pickle \
          --stderr_of_mean rms/stderr_of_mean.pickle
      cd ../../..
    done
  done
done


# Average loss in hidden state RMS experiments
cd ~/h-elmo/expres/hidden_state_rms
for ds in enwiki1G text8; do
  for opt in adam sgd; do
    for nn in 100 100_100 500_500; do
      cd ${ds}/${opt}/${nn}
      python3 ${SCRIPTS}/average_txt.py {0..19}/results/loss_valid.txt \
          -o loss_stats.txt
      cd ../../..
    done
  done
done


