import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="Path to file with input data. Input file has "
         "to contain lines in which first word is a number."
)
parser.add_argument(
    "--code",
    "-c",
    help="Code for modifying step. Has to be python "
         "expr with '{step}' substr in place where step "
         "has to be put."
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to file with output results."
)
args = parser.parse_args()


with open(args.file) as f1, open(args.output, 'w') as f2:
    for line in f1.readlines():
        words = line.split()
        step = eval(args.code.format(step=int(words[0])))
        f2.write(' '.join([str(step)] + words[1:]) + '\n')
