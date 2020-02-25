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
  "[100] enwiki1G@[100, 100] enwiki1G@[500, 500] enwiki1G@[100] \
      text8@[100, 100] text8@[500, 500] text8" \
  sgd/100@sgd/100_100@sgd/500_500@text8/sgd/100@text8/sgd/100_100@\
      text8/sgd/500_500 enwiki1G-text8/plots/sgd
unset sorting_key
unset additional_artists_str


# for sgd vary dropout loss - corr plots
cd ~/h-elmo/expres/correlation/batch/vary_dropout
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/enwiki1G/\
      noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "dropout 0.7@dropout 0.6@dropout 0.5@dropout 0.4@\
      dropout 0.3@dropout 0.2@dropout 0" \
  0.7@0.6@0.5@0.4@0.3@0.2@0 plots
unset sorting_key
unset additional_artists_str
unset lw_params


# for sgd vary dropout loss AND corr plots
cd ~/h-elmo/expres/correlation/batch/vary_dropout
sorting_key="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "dropout 0.7@dropout 0.6@dropout 0.5@dropout 0.4@\
      dropout 0.3@dropout 0.2@dropout 0" \
  0.7@0.6@0.5@0.4@0.3@0.2@0 plots
unset sorting_key


# for sgd vary learning rate loss - corr plots
cd ~/h-elmo/expres/correlation/nocorrloss/vary_lr/text8/sgd/100
lw_params=( "--linewidth" "1.0" )
sorting_key2="def sorting_key(x):
    words = x.split()
    return -float(words[-1])
"
additional_artists_str=( "-a" \
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/\
      noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "learning rate 3@learning rate 1@learning rate 0.3@learning rate 0.1@\
      learning rate 0.03@learning rate 0.01" \
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
  "learning rate 3@learning rate 1@learning rate 0.3@learning rate 0.1@\
      learning rate 0.03@learning rate 0.01" \
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
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/\
      noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "learning rate 0.01@learning rate 0.005@learning rate 0.002@\
      learning rate 0.001@learning rate 0.0003@learning rate 0.0001@\
      learning rate 0.00003@learning rate 0.00001" \
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
  "learning rate 0.01@learning rate 0.005@learning rate 0.002@\
      learning rate 0.001@learning rate 0.0003@learning rate 0.0001@\
      learning rate 0.00003@learning rate 0.00001" \
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
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/\
      noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "batch size 1024@batch size 512@batch size 256@batch size 128@batch size 64@\
      batch size 32@batch size 20@batch size 10" \
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
  "batch size 1024@batch size 512@batch size 256@batch size 128@batch size 64@\
      batch size 32@batch size 20@batch size 10" \
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
  "${EXPRES}/correlation/nocorrloss/shuffled/text8/\
      noise_best_loss_axvspan.pickle" )
sorting_key="${sorting_key2}" \
  source ${PLOT}/loss_corr_plot.sh \
  "sequence length 1000@sequence length 400@sequence length 200@\
      sequence length 100@sequence length 50@sequence length 20@\
      sequence length 10@sequence length 5" \
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
  "sequence length 1000@sequence length 400@sequence length 200@\
      sequence length 100@sequence length 50@sequence length 20@\
      sequence length 10@sequence length 5" \
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
source ${SCRIPTS}/scp_results_and_tensors_fast_2.sh correlation/batch \
    < <(gen_exp_dirs)
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


# for plotting loss and correlation 100, 100_100, 500_500
# enwiki1G BATCH ADAM - SGD
cd ~/h-elmo/expres/correlation/batch/text8
sorting_key="def sorting_key(x):
    words = x.split()
    nn = eval(' '.join(words[:-1]))
    score = 0 if words[-1] == 'adam' else 1000
    if len(nn) > 1:
        score += 100
    score += nn[0] // 10
    return score
"
source ${PLOT}/correlation_and_loss_plots.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  adam/100@adam/100_100@adam/500_500@sgd/100@sgd/100_100@sgd/500_500 plots
unset sorting_key


# for plotting loss and RMS of hidden state elements 100, 100_100, 500_500
# enwiki1G BATCH ADAM - SGD
cd ~/h-elmo/expres/hidden_state_rms
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
  source ${PLOT}/rms_and_loss_plots.sh \
  "[100] adam@[100, 100] adam@[500, 500] adam@[100] sgd@[100, 100] sgd@[500, 500] sgd" \
  enwiki1G/adam/100@enwiki1G/adam/100_100@enwiki1G/adam/500_500@enwiki1G/sgd/100@enwiki1G/sgd/100_100@enwiki1G/sgd/500_500 \
  enwiki1G/plots_adam_sgd/loss_and_rms
unset sorting_key
unset additional_artists_str


# for plotting loss and RMS of hidden state elements 100, 100_100, 500_500
# text8 BATCH ADAM - SGD
cd ~/h-elmo/expres/hidden_state_rms
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
      python3 ${SCRIPTS}/average_pickle_values.py \
          {0..19}/tensors/valid/pickle_mean_tensors/rms1.pickle \
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


# Average MNIST tensors nc-ff
cd ~/nc-ff/results
for f in ff2_adam ff2_adam_sh ff2_sgd ff2_sgd_sh; do
  mkdir ${f}/stats
  for tensor in hs0_corr hs0_rms hs1_corr hs1_rms; do

  *
    d=${f}/stats/${tensor}
    mkdir d
    python3 ${SCRIPTS}/average_pickle_values.py \
        ${f}/{0..9}/tensors/${tensor}.pickle --mean ${d}/mean.pickle \
        --stddev ${d}/stddev.pickle \
        --stderr_of_mean ${d}/stdder_of_mean.pickle \
        --preprocess "np.sqrt({array})"
  done
  python3 ${SCRIPTS}/average_txt.py ${f}/{0..9}/results/valid/loss.txt \
      -o ${f}/stats/loss.txt
done


# Average MNIST tensors nc-ff vary lr
cd ~/nc-ff/results
tensors=(hs0_corr hs0_rms hs1_corr hs1_rms hs2_corr hs2_rms hs2_corr hs2_rms \
    hs3_corr hs3_rms hs4_corr hs4_rms hs5_corr hs5_rms)
for f in ff6_adam ff6_sgd; do
    if [[ "${f}" == "ff6_adam" ]]; then
        lrs=(0.1 0.01 0.001 0.0001 0.00001 0.03 0.003 \
            0.0003 0.00003 1e-6 1e-7 3e-6 3e-7)
    elif [[ "${f}" == "ff6_sgd" ]]; then
        lrs=(0.1 0.01 0.3 0.03 1)
    else
        echo "Error! not supported experiment '${f}'. \
            Only experiments 'ff6_adam' and 'ff6_sgd' are supported." 1>&2
    fi
    for lr in "${lrs[@]}"; do
        stats="${f}/lr${lr}/stats"
        mkdir "${stats}"
        for tensor in "${tensors[@]}"; do
            d=${f}/lr${lr}/stats/${tensor}
            mkdir "${d}"
            python3 ${SCRIPTS}/average_pickle_values.py \
                "${f}/lr${lr}/"{0..9}"/tensors/${tensor}.pickle" \
                --mean ${d}/mean.pickle \
                --stddev ${d}/stddev.pickle \
                --stderr_of_mean ${d}/stdder_of_mean.pickle \
                --preprocess "np.sqrt({array})"
        done
        python3 ${SCRIPTS}/average_txt.py \
            "${f}/lr${lr}/"{0..9}"/results/valid/loss.txt" \
            -o "${stats}/loss.txt"
    done
done


# Draw nc-ff plots
cd ~/nc-ff/results
mkdir plots
step_file=ff2_adam/0/results/valid/loss.txt
tensors=(hs0_corr hs0_rms hs1_corr hs1_rms)
ylabels=("mean square correlation" "mean square element" \
    "mean square correlation" "mean square element")
for i in {0..3}; do
    python3 ${PLOT}/plot_data_from_pickle.py -l adam sgd \
        -s "${step_file}" "${step_file}" \
        -m "ff2_adam/stats/${tensors[i]}/mean.pickle" \
        "ff2_sgd/stats/${tensors[i]}/mean.pickle" \
        -d "ff2_adam/stats/${tensors[i]}/stddev.pickle" \
        "ff2_sgd/stats/${tensors[i]}/stddev.pickle" -n \
        -o "plots/adam-sgd/${tensors[i]}_plot_data.pickle"
    python3 ${PLOT}/plot_from_pickle.py \
        "plots/adam-sgd/${tensors[i]}_plot_data.pickle" \
        -x step -y "${ylabels[i]}" -X symlog \
        -o "plots/adam-sgd/${tensors[i]}_plot" \
        -t fill -d best -O -s png -r 900 -g -w both
done
python3 ${PLOT}/plot_data_from_txt.py ff2_adam/stats/loss.txt \
    ff2_sgd/stats/loss.txt -l adam sgd -x 0 -y 1 -e 2 \
    -o plots/adam-sgd/loss_plot_data.pickle -n
python3 ${PLOT}/plot_from_pickle.py plots/adam-sgd/loss_plot_data.pickle \
    -x step -y loss -X symlog -o plots/adam-sgd/loss_plot -t fill -d best -O \
    -s png -r 900 -g -w both


# Draw nc-ff plots vary lr
cd ~/nc-ff/results
mkdir plots
step_file=ff6_adam/lr0.1/0/results/valid/loss.txt
tensors=(hs0_corr hs0_rms hs1_corr hs1_rms hs2_corr hs2_rms \
    hs3_corr hs3_rms hs4_corr hs4_rms hs5_corr hs5_rms)
ylabels=("mean square correlation" "mean square element")
sorting_key="def sorting_key(x):
    return float(x.split()[-1])
"
sorting_key_script="ff6_adam/plots/hs0_corr/data_exec.py"
for f in ff6_adam ff6_sgd; do
    if [[ "${f}" == "ff6_adam" ]]; then
        lrs=(0.001 0.0001 0.00001 \
            0.0003 0.00003 1e-6 1e-7 3e-6 3e-7)
    elif [[ "${f}" == "ff6_sgd" ]]; then
        lrs=(0.1 0.01 0.3 0.03 1)
    else
        echo "Error! not supported experiment '${f}'. \
            Only experiments 'ff6_adam' and 'ff6_sgd' are supported." 1>&2
    fi

    labels=()
    step_files=()
    loss_files=()
    for lr in "${lrs[@]}"; do
        stats="${f}/lr${lr}/stats"
        labels+=("learning rate ${lr}")
        step_files+=("${step_file}")
        loss_files+=("${stats}/loss.txt")
    done
    for i in {0..11}; do
        let j=i%2
        mean_files=()
        stddev_files=()
        for lr in "${lrs[@]}"; do
            stats="${f}/lr${lr}/stats"
            mean_files+=("${stats}/${tensors[i]}/mean.pickle")
            stddev_files+=("${stats}/${tensors[i]}/stddev.pickle")
        done

        python3 ${PLOT}/plot_data_from_pickle.py -l "${labels[@]}" \
            -s "${step_files[@]}" -m "${mean_files[@]}" \
            -d "${stddev_files[@]}" -n \
            -o "${f}/plots/${tensors[i]}/data.pickle" -k="${sorting_key}"
        python3 ${PLOT}/plot_from_pickle.py \
            "${f}/plots/${tensors[i]}/data.pickle" \
            -x step -y "${ylabels[j]}" -X symlog \
            -o "${f}/plots/${tensors[i]}/plot" \
            -t fill -d best -O -s png -r 900 -g -w both \
            -e "${sorting_key_script}"
    done
    python3 ${PLOT}/plot_data_from_txt.py "${loss_files[@]}" \
        -l "${labels[@]}" -x 0 -y 1 -e 2 \
        -o "${f}/plots/loss/data.pickle" -n -k="${sorting_key}"
    python3 ${PLOT}/plot_from_pickle.py \
        "${f}/plots/loss/data.pickle" -x step -y loss -X symlog -o \
        "${f}/plots/loss/plot" -t fill -d best -O -s png -r 900 -g -w both \
        -e "${sorting_key_script}"
done


# Draw long dropout plots
cd ~/h-elmo/expres/correlation/batch/long_dropout
mkdir plots
step_file=0/0/results/loss_valid.txt
declare -a labels steps corrs losses
for dp in 0 0.4 0.7; do
  labels+=("dropout ${dp}")
  steps+=("${dp}/0/results/loss_valid.txt")
  corrs+=("${dp}/0/tensors/valid/pickle_mean_tensors/correlation.pickle")
  losses+=("${dp}/0/results/loss_valid.txt")
done
python3 ${PLOT}/plot_data_from_pickle.py -l "${labels[@]}" -s "${steps[@]}" \
  -m "${corrs[@]}" -n -o plots/corr_data.pickle -p sqrt
python3 ${PLOT}/plot_data_from_txt.py "${steps[@]}" -l "${labels[@]}" -x 0 \
    -y 1 -o plots/loss_data.pickle -n
python3 ${PLOT}/plot_from_pickle.py plots/corr_data.pickle \
    -x step -y "mean square correlation" -X symlog -o "plots/corr_plot" \
    -t noerr -d best -O -s png -r 900 -g -w both
python3 ${PLOT}/plot_from_pickle.py plots/loss_data.pickle \
    -x step -y loss -X symlog -o plots/loss_plot -t noerr -d best -O \
    -s png -r 900 -g -w both


# Fix histograms for entropy and mutual information
cd /media/anton/DATA/results/h-elmo/expres/entropy/first_experiment/hist
path=tensors/valid/accumulator_postprocessing
for i in {4..9}; do
  for h in "${i}/${path}/"*; do
    python3 "${SCRIPTS}/fix_no_hist_reset.py" \
      "${h}" "${h%.*}_fixed.pickle"
    rm "${h}"
    mv "${h%.*}_fixed.pickle" "${h}"
  done
done


# Compute entropy and mutual information
cd /media/anton/DATA/results/h-elmo/expres/entropy/first_experiment/hist
path=tensors/valid/accumulator_postprocessing
for i in {4..9}; do
  launch_path="${i}/${path}"
  for h in "${launch_path}/"hist_*; do
    h_name=$(basename -- "${h}")      # histogram file name
    hs_name="${h_name#hist}"          # hidden state name
    cr_h_name="cross_hist${hs_name}"  # cross histogram file name

    e_name="entropy${hs_name}"
    mean_e_name="mean_${e_name}"
    e_path="${launch_path}/${e_name}"
    mean_e_path="${launch_path}/${mean_e_name}"

    mi_name="mi${hs_name}"
    mean_mi_name="mean_${mi_name}"
    mi_path="${launch_path}/${mi_name}"
    mean_mi_path="${launch_path}/${mean_mi_name}"

    python3 "${SCRIPTS}/hist2entropy.py" "${h}" "${e_path}"
    python3 "${SCRIPTS}/hist2mi.py" "${h}" "${launch_path}/${cr_h_name}" \
      "${mi_path}"
    python3 "${SCRIPTS}/array_mean.py" "${e_path}" "${mean_e_path}"
    python3 "${SCRIPTS}/array_mean.py" "${mi_path}" "${mean_mi_path}"
  done
done


# Average correlation for the second layer of LSTM
cd ~/h-elmo/expres/correlation/batch/second_layer
for dp in dp0 dp0.7; do
  python3 "${SCRIPTS}/average_pickle_values.py" \
      "${dp}"/{0..19}/tensors/valid/pickle_mean_tensors/correlation2.pickle \
      --mean "${dp}"/mean/corr2/mean.pickle \
      --stddev "${dp}"/mean/corr2/stddev.pickle \
      --stderr_of_mean "${dp}"/mean/corr2/stderr_of_mean.pickle \
      --preprocess "np.sqrt({array})"
  python3 "${SCRIPTS}/average_pickle_values.py" \
      "${dp}"/{0..19}/tensors/valid/pickle_mean_tensors/correlation12.pickle \
      --mean "${dp}"/mean/corr12/mean.pickle \
      --stddev "${dp}"/mean/corr12/stddev.pickle \
      --stderr_of_mean "${dp}"/mean/corr12/stderr_of_mean.pickle \
      --preprocess "np.sqrt({array})"
  python3 "${SCRIPTS}/average_txt.py" "${dp}"/{0..19}/results/loss_valid.txt \
      --output "${dp}/mean/loss.txt"
done


# Compare correlation on the first and the second layers
cd ~/h-elmo/expres/correlation/batch/second_layer/dp0
python3 "${PLOT}/plot_data_from_pickle.py" -l "layer 1" "layer 2" \
  -s ../../vary_dropout/0/0/results/loss_valid.txt 0/results/loss_valid.txt \
  -m ../../vary_dropout/0/mean/corr/mean.pickle mean/corr2/mean.pickle \
  -d ../../vary_dropout/0/mean/corr/stddev.pickle mean/corr2/stddev.pickle \
  -o plots/comp_1st_2nd.pickle
python3 "${PLOT}/plot_from_pickle.py" plots/comp_1st_2nd.pickle \
  -y "mean square correlation" -X symlog -t fill -d best -g -w both \
  -s png -o plots/comp_1st_2nd


# Entropy and mutual information
cd ~/h-elmo/expres/entropy/first_experiment/hist
declare -a ent_files_1
declare -a mi_files_1
declare -a ent_files_2
declare -a mi_files_2
for i in {0..7}; do
  path="${i}/tensors/valid/accumulator_postprocessing"
  ent_files_1+=("${path}/mean_entropy_level0_0_hidden_state.pickle")
  mi_files_1+=("${path}/mean_mi_level0_0_hidden_state.pickle")
  ent_files_2+=("${path}/mean_entropy_level0_1_hidden_state.pickle")
  mi_files_2+=("${path}/mean_mi_level0_1_hidden_state.pickle")
done
python3 "${SCRIPTS}/average_pickle_values.py" \
    "${ent_files_1[@]}" --mean mean/entropy1/mean.pickle \
    --stddev mean/entropy1/stddev.pickle \
    --stderr_of_mean mean/entropy1/stderr_of_mean.pickle
python3 "${SCRIPTS}/average_pickle_values.py" \
    "${mi_files_1[@]}" --mean mean/mi1/mean.pickle \
    --stddev mean/mi1/stddev.pickle \
    --stderr_of_mean mean/mi1/stderr_of_mean.pickle
python3 "${SCRIPTS}/average_pickle_values.py" \
    "${ent_files_2[@]}" --mean mean/entropy2/mean.pickle \
    --stddev mean/entropy2/stddev.pickle \
    --stderr_of_mean mean/entropy2/stderr_of_mean.pickle
python3 "${SCRIPTS}/average_pickle_values.py" \
    "${mi_files_2[@]}" --mean mean/mi2/mean.pickle \
    --stddev mean/mi2/stddev.pickle \
    --stderr_of_mean mean/mi2/stderr_of_mean.pickle
python3 "${PLOT}/plot_data_from_pickle.py" -l "layer 1" "layer 2" \
    -s 0/results/loss_valid.txt 0/results/loss_valid.txt \
    -m mean/entropy1/mean.pickle mean/entropy2/mean.pickle \
    -d mean/entropy1/stddev.pickle mean/entropy2/stddev.pickle \
    -o plots/entropy.pickle
python3 "${PLOT}/plot_data_from_pickle.py" -l "layer 1" "layer 2" \
    -s 0/results/loss_valid.txt 0/results/loss_valid.txt \
    -m mean/mi1/mean.pickle mean/mi2/mean.pickle \
    -d mean/mi1/stddev.pickle mean/mi2/stddev.pickle \
    -o plots/mi.pickle
python3 "${PLOT}/plot_from_pickle.py" plots/entropy.pickle \
    -y "entropy, bits" -X symlog -t fill -d best -g -w both -s png \
    -o plots/entropy
python3 "${PLOT}/plot_from_pickle.py" plots/mi.pickle \
    -y "mutual information, bits" -X symlog -t fill -d best -g -w both \
    -s png -o plots/mi --bottom 0


# Eigen values of kernels of feedforward neural networks
cd ~/nc-ff/results/eigen/no_reg/truncated_normal/

tensors=(kernel0 kernel1 kernel2)
stats="stats"
mkdir "${stats}"
for tensor in "${tensors[@]}"; do
    d="stats/${tensor}"
    mkdir "${d}"
    python3 ${SCRIPTS}/average_pickle_values.py \
        {0..9}"/tensors/${tensor}.pickle" --mean ${d}/mean.pickle \
        --stddev ${d}/stddev.pickle \
        --stderr_of_mean ${d}/stdder_of_mean.pickle \
        --preprocess "np.sqrt({array})"
done
python3 ${SCRIPTS}/average_txt.py {0..9}"/results/valid/loss.txt" \
    -o "${stats}/loss.txt"

cd ~/nc-ff/results/eigen/no_reg/truncated_normal/
mkdir plots
step_file=0/results/valid/loss.txt
tensors=(kernel0 kernel1 kernel2)
ylabel="fraction of real eigen values"
for i in {0..2}; do
    python3 ${PLOT}/plot_data_from_pickle.py -l first -s "${step_file}" \
        -m "stats/${tensors[i]}/mean.pickle" \
        -d "stats/${tensors[i]}/stddev.pickle" \
        -o "plots/${tensors[i]}/step/data.pickle"
    python3 ${PLOT}/plot_from_pickle.py \
        "plots/${tensors[i]}/step/data.pickle" \
        -x step -y "${ylabel}" -X symlog \
        -o "plots/${tensors[i]}/step/plot" \
        -t fill -d None -O -s png -r 900 -g -w both
done

python3 ${PLOT}/plot_data_from_txt.py stats/loss.txt -l first -x 0 -y 1 -e 2 \
    -o plots/loss/data.pickle
python3 ${PLOT}/plot_from_pickle.py plots/loss/data.pickle -x step -y loss \
    -X symlog -o plots/loss/plot -t fill -d None -O -s png -r 900 -g \
    -w both
