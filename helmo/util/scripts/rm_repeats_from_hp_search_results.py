import argparse
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help='A path to file with test results of a hyper parameter search.'
    )
    parser.add_argument(
        'output',
        help='A path to file where filtered results are stored.'
    )
    return parser.parse_args()


def main():
    args = get_args()
    unique = OrderedDict()
    with open(args.input) as f:
        for line in f:
            words = line.split()
            key = ' '.join(words[4:])
            if key not in unique:
                unique[key] = line
    with open(args.output, 'w') as f:
        for line in unique.values():
            f.write(line)


if __name__ == '__main__':
    main()
