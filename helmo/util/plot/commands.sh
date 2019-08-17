#!/usr/bin/env bash

# for plotting loss and correlation in 100 100_100 500_500
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/correlation_and_loss_plots.sh "[100]@[100, 100]@[500, 500]" 100@100_100@500_500 plots

# for plotting wide correlation
cd ~/h-elmo/expres/correlation/nocorrloss
source ${PLOT}/correlation_and_loss_plots.sh "dropout 0@dropout 0.2@dropout 0.4@dropout 0.7" \
  wide/0@wide/0.2@wide/0.4@wide/0.7 wide/plots/20

# for plotting adam 100 100_100 500_500
cd ~/h-elmo/expres/correlation/nocorrloss
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/correlation_and_loss_plots.sh "[100]@[100, 100]@[500, 500]" \
  adam/100@batch_mean/100_100short@wide/0 adam/plots

# for plotting loss - correlation 100 100_100 500_500
sorting_key="def sorting_key(x):"$'\n'"    return tuple(eval(x))" \
  source ${PLOT}/loss_corr_plot.sh "[100]@[100, 100]@[500, 500]" 100@100_100@500_500 plots

# for plotting loss - correlation 100 100_100 500_500 for 2 optimizers text8
# ADD `-S 1.5 3.5` TO plot_from_pickle.py call
cd ~/h-elmo/expres/correlation/nocorrloss/text8
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