import argparse
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='A path to a file with hp search results.'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='A path to a csv file with results.',
        default='num_nodes.csv'
    )
    parser.add_argument(
        '--metrics',
        '-m',
        help='A list of names of metrics which will be added to csv file.',
        nargs='+',
        default=['loss'],
    )
    return parser.parse_args()


def main():
    args = get_args()
    output = []
    launches_nodes = []
    launches_metrics = {m: [] for m in args.metrics}
    num_launches = 0
    with open(args.input) as f:
        inp_header = f.readline().split()
        metrics_indices_in_header = {
            m: i for i, m in enumerate(inp_header[:4])}
        for line in f:
            words = line.split()
            nodes = ' '.join(words[4:])
            nodes = nodes[1:-1].split(', ')
            launches_nodes.append(nodes)
            for m in args.metrics:
                launches_metrics[m].append(words[metrics_indices_in_header[m]])
            num_launches += 1
    num_layers = max([len(nn) for nn in launches_nodes])
    output_header = ['layer {}'.format(i+1) for i in range(num_layers)] + \
        args.metrics
    output.append(output_header)
    for i in range(num_launches):
        mvalues = [launches_metrics[m][i] for m in args.metrics]
        output.append(launches_nodes[i] + mvalues)
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)


if __name__ == '__main__':
    main()
