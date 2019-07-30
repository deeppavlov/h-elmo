#!/usr/bin/env bash
pwd
for i in {0..9}
do
  echo ""
  echo "  $i"
  for tag in $tags
  do
    echo $tag
    mkdir -p $i/plots/correlations $i/plots/correlations_relative_to_match_stddevs
    python3 $HH/util/plot/hist.py $i/*/$tag/correlations.pickle --labels $labels -o $i/plots/correlations/$tag --lgd upper_left -d
    for layer in $layers
    do
      python3 $HH/util/scripts/apply_expr_to_pickle.py $i/$layer/$tag/correlations.pickle $i/$layer/$tag/match_stddevs.pickle -c "np.abs({0})/{1}" -o $i/$layer/$tag/correlations_relative_to_match_stddevs.pickle
    done
    python3 $HH/util/plot/hist.py $i/*/$tag/correlations_relative_to_match_stddevs.pickle --labels $labels -o $i/plots/correlations_relative_to_match_stddevs/$tag --lgd upper_left -d
  done
done
