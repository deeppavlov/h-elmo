import argparse
import json

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="Path to file with results.",
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to file with average results",
    default="mean.json",
)
args = parser.parse_args()


with open(args.file) as f:
    lines = f.readlines()

headers = lines[0].split()

metrics = headers[:4]
hps = headers[4:]

results = {}
for line in lines[1:]:
    values = line.split()
    metric_values = tuple(map(float, values[:4]))
    hp_values = tuple(map(float, values[4:]))
    if hp_values in results:
        for m, mv in zip(metrics, metric_values):
            results[hp_values][m].append(mv)
    else:
        results[hp_values] = {}
        for m, mv in zip(metrics, metric_values):
            results[hp_values][m] = [mv]

for v in results.values():
    for m in v:
        v[m] = {'mean': np.mean(v[m]), 'std': np.std(v[m], ddof=1)}

results_json = []
for hpv, stats in results.items():
    one_comb_results = dict(
        hp=list(hpv)
    )
    for m, mv in stats.items():
        one_comb_results[m] = mv
    results_json.append(one_comb_results)

output = dict(
    order=hps,
    results=results_json,
)

with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)
