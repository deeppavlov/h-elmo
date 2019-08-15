import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="Path to pickle file."
)
args = parser.parse_args()

n = 0
with open(args.file, 'rb') as f:
    while True:
        try:
            _ = pickle.load(f)
        except EOFError:
            break
        n += 1

print(args.file, n)
