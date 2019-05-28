import argparse

import helmo.util.path_help as path_help


parser = argparse.ArgumentParser()
parser.add_argument(
    "paths",
    help="A list of files from which labels are generated",
    nargs='+',
)
parser.add_argument(
    "--output_separator",
    "-s",
    help="Labels are returned as string in which they are separated "
         "by `output_separator`. Default is whitespace.",
    default=' ',
)
args = parser.parse_args()


labels = path_help.labels_from_paths(args.paths)

print(args.output_separator.join(labels))
