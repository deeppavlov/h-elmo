import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_name",
        help="Path to file with results",
    )
    parser.add_argument(
        "step",
        help="Step from which results are taken.",
        type=int,
    )
    args = parser.parse_args()
    return args


def get_step_and_value(line):
    s, v = line.split()
    return int(s), float(v)


def get_value_on_step(file_name, step):
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        st, value = get_step_and_value(line)
        if st == step:
            return value


def main():
    args = get_args()
    value = get_value_on_step(args.file_name, args.step)
    print(value)


if __name__ == '__main__':
    main()
