import sys, os
import random
import argparse
import itertools

random.seed(0)


def get_attr(fname):
    if len(fname.split()) > 1:
        attr = float(fname.split()[1])
    else:
        attr = float(fname.split('_')[0])
    return attr


def get_pair_label(attrA, attrB, margin):
    if abs(attrA-attrB) <= margin:
        return 1
    elif attrA < attrB:
        return 0
    else:
        return 2


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt', help='txt file')
parser.add_argument('--output', type=str, default='output.txt', help='base name for output file')
parser.add_argument('--num_pairs', type=int, default=10000)
parser.add_argument('--margin', type=float, default=10)
opt = parser.parse_args()
opt.baseroot = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)

# list all files
with open(opt.input, 'r') as f:
    lines = [line.rstrip('\n') for line in f.readlines()]

# all possible pairs, shuffled
all_pairs = []
if len(lines) <= 10000:
    all_combinations = list(itertools.combinations(range(len(lines)), 2))
    for pair in all_combinations:
        if random.random() < 0.5:
            pair = pair[::-1]
        all_pairs.append(pair)
    random.shuffle(all_pairs)
else:
    # random sample
    indice = range(len(lines))
    for _ in range(opt.num_pairs):
        all_pairs.append(random.sample(indice, 2))

with open(opt.output, 'w') as f:
    for pair in all_pairs[:opt.num_pairs]:
        fileA = lines[pair[0]]
        fileB = lines[pair[1]]
        label = get_pair_label(get_attr(fileA), get_attr(fileB), opt.margin)
        f.write('%s %s %d\n' % (fileA.split()[0], fileB.split()[0], label))
