import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    "-f",
    help="List of text files with results.",
    nargs='+',
    type=argparse.FileType(mode='r'),
)
parser.add_argument(
    "--min",
    "-m",
    help="If provided script will minimum value. Else it will find "
         "maximum value.",
    action="store_true",
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to output file. If not provided result is printed "
         "stdout."
)
args = parser.parse_args()


if args.min:
    m = float('inf')
else:
    m = float('-inf')

for f in args.files:
    v = float(f.read())
    if not args.min ^ (v < m):
        m = v

if args.output is None:
    print(m)
else:
    with open(args.output, 'w') as f:
        f.write(str(m) + '\n')
