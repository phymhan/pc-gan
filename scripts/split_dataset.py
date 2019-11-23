import sys, os
import random
import argparse
import itertools
import math
import numpy

random.seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt', help='txt file')
parser.add_argument('--output', type=str, default='output.txt', help='base name for output file')
parser.add_argument('--num_samples', nargs='*', type=int, default=[10000, 1000, -1], help='number of samples for TRAIN, VAL, TEST, respectively')
opt = parser.parse_args()
opt.baseroot = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)

with open(opt.input, 'r') as f:
    lines = [line.rstrip('\n') for line in f.readlines()]

num_total = len(lines)
if opt.num_samples[-1] == -1:
    opt.num_samples[-1] = num_total-sum(opt.num_samples[0:-1])

random.shuffle(lines)
idx_prev = 0
for mode, idx in zip(['train', 'val', 'test'], numpy.cumsum(opt.num_samples)):
    subset = range(idx_prev, idx)
    idx_prev = idx
    if len(subset) == 0:
        continue
    with open(opt.output.rstrip('.txt')+'_%s'%mode+'.txt', 'w') as f:
        for line_idx in subset:
            f.write(lines[line_idx]+'\n')
