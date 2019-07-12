import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "file1",
    help="First file (file with pretraining process)",
)
parser.add_argument(
    "file2",
    help="Second file (file with posttraining results)",
)
parser.add_argument(
    "ckpt_dir",
    help="Directory with best checkpoint best score and best step",
)
parser.add_argument(
    "output",
    help="Output file",
)
args = parser.parse_args()


step_file = os.path.join(args.ckpt_dir, 'best_step.txt')
score_file = os.path.join(args.ckpt_dir, 'best_best_loss_on_valid.txt')
with open(step_file) as f:
    init_step = int(f.read())
# print(init_step)


def get_step_from_result_line(line):
    if line:
        words = line.split()
        return int(words[0])


def get_score_from_result_line(line):
    if line:
        words = line.split()
        return float(words[1])


def slice_results_by_step(text, step1, step2):
    lines = text.strip().split('\n')
    result = []
    for line in lines:
        if line:
            s = get_step_from_result_line(line)
            if step1 <= s < step2:
                result.append(line)
    return '\n'.join(result) + '\n'
    

with open(args.file1) as f:
    text = f.read()
text = slice_results_by_step(text, 0, init_step+1)
lines = text.strip().split('\n')

assert get_step_from_result_line(lines[-1]) == init_step, "last step in text is {} but best step is {}".format(get_step_from_result_line(lines[-1]), init_step) 
best_score = get_score_from_result_line(lines[-1])

with open(args.file2) as f:
    for line in f.readlines():
        words = line.split()
        step = int(words[0])
        step += init_step
        other = ' '.join(words[1:])
        text += str(step) + ' ' + other + '\n'

      
with open(args.output, 'w') as f:
    f.write(text)

